import functools
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import backoff
import cloudpickle
import joblib.memory
from cloudpickle.compat import pickle
from joblib.func_inspect import filter_args, get_func_code, get_func_name
from joblib.hashing import hash
from joblib.memory import extract_first_line


def get_cached_func_location(func):
    """Find the location inside of your <cache>/joblib/ folder where a cached function
    is stored. Necessary because each function will have multiple subcaches for its
    codebase.

    Args:
        func (Callable): a function that has been cached

    Returns:
        str: the path where the input function is stored
    """
    return joblib.memory._build_func_identifier(func.func)


def naive_func_identifier(func):
    """Build simple identifier based on function name

    Args:
        func (Callable): a function to cache

    Returns:
        str: a string identifying the input function based on its location in the
        import hierarchy
    """
    modules, funcname = get_func_name(func)
    modules.append(funcname)
    return modules


def better_build_func_identifier(func):
    """Build a roughly unique identifier for the cached function.

    Args:
        func (Callable): a function to cache

    Returns:
        tuple[str]: a list of components identifying a function based on its code and
        location in the import hierarchy
    """
    parts = []
    parts.extend(naive_func_identifier(func))
    func_id, h_func_id, h_code = hash_func(func)
    parts.append(str(h_code))

    # We reuse historical fs-like way of building a function identifier
    return tuple(parts)


def token(config) -> str:
    """Generates a unique config identifier. Taken from stackoverflow 45674572.

    Args:
        config (dict): a catlas input config

    Returns:
        str: A hex token identifying the config.
    """
    # `sign_mask` is used to make `hash` return unsigned values
    sign_mask = (1 << sys.hash_info.width) - 1
    # Use `json.dumps` with `repr` to ensure the config is hashable
    json_config = json.dumps(config, default=repr)
    # Get the hash as a positive hex value with consistent padding without '0x'
    return f"{hash(json_config) & sign_mask:#0{sys.hash_info.width//4}x}"[2:]


def hash_func(func):
    """Hash the function id, its file location, and the function code.

    Args:
        func (Callable): a function to cache

    Returns:
        str: a hash uniquely identifying the function
    """
    func_code, _, first_line = get_func_code(func)
    func_code_h = hash([func_code, first_line])
    return id(func), hash(os.path.join(*naive_func_identifier(func))), func_code_h


def check_cache(cached_func):
    """checks if cached function is safe to call without overriding cache (adapted from https://github.com/joblib/joblib/blob/7742f5882273889f7aaf1d483a8a1c72a97d57e3/joblib/memory.py#L672)

    Inputs:
        cached_func (Callable): Function to check cache for

    Returns:
        bool: True if cached function is safe to call, else False

    """

    """Here, we go through some effort to be robust to dynamically
    changing code and collision. We cannot inspect.getsource
    because it is not reliable when using IPython's magic "%run"."""
    func_code, source_file, first_line = cached_func.func_code_info
    func_id = joblib.memory._build_func_identifier(cached_func.func)

    try:
        old_func_code, old_first_line = extract_first_line(
            cached_func.store_backend.get_cached_func_code([func_id])
        )
    except (IOError, OSError):  # code has not been written
        # cached_func._write_func_code(func_code, first_line)
        return True
    if old_func_code == func_code:
        return True

    return False


NO_CACHE_RESULT = "not a cache result"


def sqlitedict_memoize(
    folder,
    func,
    ignore=(),
    coerce_mmap=False,
    shard_digits=2,
):
    """cache functions in a way that splits cached functions between many folders.

    Args:
        folder (str): file location where cache shoud be created
        func (Callable): function to cache
        ignore (tuple[str], optional): List of arguments that will be ignored when determining whether to start a new cache. Defaults to ().
        coerce_mmap (bool, optional): if True, don't distinguish between numpy ndarrays and numpy memmaps. Defaults to False.
        shard_digits (int, optional): Generate 16^(shard digits) different folders to store functions in. Defaults to 2.

    Returns:
        Callable: cached function
    """
    # Use a base name that includes the full function path and a hash on the function
    # code itself
    base = better_build_func_identifier(func)

    @backoff.on_exception(
        backoff.expo, (sqlite3.OperationalError, TimeoutError), max_time=180
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper for callable to cache arguments and return values. If the function has been called with these inputs already, pull the result from cache; otherwise, compute it and store the result.

        Returns:
            Any: the result of the function call.
        """ """"""

        # Get the key for the arguments
        key = hash(
            filter_args(func, ignore, args, kwargs), coerce_mmap=coerce_mmap
        ).encode("utf-8")

        # Generate a base db location based on the full function path and a hash on the
        # function code itself
        db_loc = ".".join(base)

        # Add a couple of shard digits so that actually we reference a subfolder
        # (string is hex, so two shard digits is 16^2 shards)
        # the numer of shards should be approximately equal to the number of expected
        # simultaneous writers
        if shard_digits > 0:
            db_loc = str(db_loc) + "/" + key.decode()[0:shard_digits] + ".sqlite"

        result = NO_CACHE_RESULT

        cache = SqliteSingleThreadDict(
            folder + "/" + db_loc,
        )

        # Grab the cached entry (might be None)
        # if key in cache:
        try:
            cache.__enter__()
            result = cache[key]
        except (pickle.PicklingError, sqlite3.OperationalError, KeyError):
            # It's in the cache, but the pickle data is corrupted :(
            pass

        # If we need to compute the result, do it, and cache the result
        if (type(result) == str) & (result == NO_CACHE_RESULT):
            result = func(*args, **kwargs)
            cache[key] = result

        # Tidy up!
        cache.close()
        del cache

        return result

    return wrapper


class SqliteSingleThreadDict(dict):
    """A dictionary connected to a sqlite database.
        Acts like a dictionary, but is actually a file that opens file connections when items are accessed or set.

    Raises:
        KeyError: an accessed key didn't exist in the dictionary.

    """

    # This code was originally adapted from the excellect sqlitedict package (Apache2
    # license)
    # It was almost entirely re-written as the use case here is much simpler than what
    # sqlitedict provides

    def __init__(
        self,
        filename=None,
        tablename="unnamed",
        journal_mode="PERSIST",
        encode=cloudpickle.dumps,
        decode=cloudpickle.loads,
    ):
        self.filename = filename
        self.journal_mode = journal_mode
        self.encode = encode
        self.decode = decode
        self.tablename = tablename

    def _new_readonly_conn(self):
        """Open a read-only (ro) connection to the database.
            This is used to either check if data exists, or retrive cached data

        Returns:
            Connection: a read-only connection to a file
        """
        # This function opens a read-only (ro) connection to the database
        # which is used to either check if data exists, or retrive cached data
        try:
            conn = sqlite3.connect(
                f"file:{self.filename}?mode=ro",
                check_same_thread=True,
                timeout=1,
                uri=True,
            )
        except sqlite3.OperationalError:
            # If we hit an error in a read-only open, it probably means that
            # the database does not actually exist, so we should make it
            Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                f"file:{self.filename}?mode=ro",
                check_same_thread=True,
                timeout=1,
                uri=True,
            )
        conn.isolation_level = None
        return conn

    def _new_readwrite_conn(self):
        """Open a read/write connection.
            This is necessary to insert data into the cache

        Returns:
            Connection: a read-write connection to a file
        """
        try:
            conn = sqlite3.connect(self.filename, check_same_thread=True, timeout=30)
        except sqlite3.OperationalError:
            # If we hit an error, it's probably because the directory doesn't exist,
            # so make it first!
            Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.filename, check_same_thread=True, timeout=30)
        conn.isolation_level = None

        # Set some settings for the DB/connection and make the table if needed
        with conn:
            conn.execute("PRAGMA journal_mode = %s" % self.journal_mode)
            conn.execute("PRAGMA synchronous=FULL")
            MAKE_TABLE = (
                'CREATE TABLE IF NOT EXISTS "%s" (key TEXT PRIMARY KEY, value BLOB)'
                % self.tablename
            )
            conn.execute(MAKE_TABLE)

        conn.commit()
        return conn

    def __enter__(self):
        if not hasattr(self, "conn") or self.conn is None:
            self.readonly_conn = self._new_readonly_conn()
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        """Close read-only connection"""
        if hasattr(self, "readonlyo_conn") and self.readonly_conn is not None:
            self.readonly_conn.close()
            self.readonly_conn = None

    def __contains__(self, key):
        """Check if item is in sqlite dictionary already.

        Args:
            key (str): a key to check

        Returns:
            bool: True if key is in the sqlite dictionary.
        """
        HAS_ITEM = 'SELECT 1 FROM "%s" WHERE key = ?' % self.tablename
        with self.readonly_conn as conn:
            return len(conn.execute(HAS_ITEM, (key,)).fetchall()) > 0

    def __getitem__(self, key):
        """Read an item from the cache given a key.

        Args:
            key (str): key to read

        Raises:
            KeyError: key not found

        Returns:
            Any: value accessed at key
        """
        with self.readonly_conn as readonly_conn:
            GET_ITEM = 'SELECT value FROM "%s" WHERE key = ?' % self.tablename
            item = readonly_conn.execute(GET_ITEM, (key,)).fetchall()
        if len(item) == 0:
            raise KeyError(key)
        (value,) = item[0]
        return self.decode(value)

    def __setitem__(self, key, value):
        """Set an item in the cache given a key/value pair

        Args:
            key (str): key to set
            value (Any): object to store
        """
        with closing(self._new_readwrite_conn()) as readwrite_conn:
            ADD_ITEM = 'REPLACE INTO "%s" (key, value) VALUES (?,?)' % self.tablename
            readwrite_conn.execute(ADD_ITEM, (key, self.encode(value)))
            readwrite_conn.commit()
