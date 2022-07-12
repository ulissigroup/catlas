import os
import joblib.memory
from joblib.func_inspect import get_func_name
from joblib.memory import (
    extract_first_line,
    JobLibCollisionWarning,
)
import hashlib as hl
import sys
import json
from joblib.func_inspect import get_func_code, filter_args
from joblib.hashing import hash
import cloudpickle
import lmdb
import time
from cloudpickle.compat import pickle
import sqlite3
import numpy as np
from contextlib import closing

def get_cached_func_location(func):
    """Find the location inside of your <cache>/joblib/ folder where a cached function is stored.
    Necessary because each function will have multiple subcaches for its codebase."""
    return joblib.memory._build_func_identifier(func.func)


def naive_func_identifier(func):
    """Build simple identifier based on function name"""
    modules, funcname = get_func_name(func)
    modules.append(funcname)
    return modules


def better_build_func_identifier(func):
    """Build a roughly unique identifier for the cached function."""
    parts = []
    parts.extend(naive_func_identifier(func))
    func_id, h_func_id, h_code = hash_func(func)
    parts.append(str(h_code))

    # We reuse historical fs-like way of building a function identifier
    return tuple(parts)


def token(config) -> str:
    """Generates a hex token that identifies a config.
    taken from stackoverflow 45674572
    """
    # `sign_mask` is used to make `hash` return unsigned values
    sign_mask = (1 << sys.hash_info.width) - 1
    # Use `json.dumps` with `repr` to ensure the config is hashable
    json_config = json.dumps(config, default=repr)
    # Get the hash as a positive hex value with consistent padding without '0x'
    return f"{hash(json_config) & sign_mask:#0{sys.hash_info.width//4}x}"[2:]


def hash_func(func):
    """Hash the function id, its file location, and the function code"""
    func_code, _, first_line = get_func_code(func)
    func_code_h = hash([func_code, first_line])
    return id(func), hash(os.path.join(*naive_func_identifier(func))), func_code_h


class CacheOverrideError(Exception):
    """Exception raised for function calls that would wipe an existing cache

    Attributes:
        memorized_func -- cached function that raised the error
    """

    def __init__(
        self,
        cached_func,
        message="Existing cache would be overridden: %s\nPlease revert your copy of this function to look like the code in the existing cache OR start a new cache OR backup/delete the existing cache manually",
    ):
        self.cached_func = cached_func
        self.message = message
        super().__init__(
            self.message
            % os.path.join(
                cached_func.store_backend.location,
                joblib.memory._build_func_identifier(cached_func.func),
            )
        )


def safe_cache(memory, func, *args, **kwargs):
    """Wrapper for memory.cache(func) that raises an error if the cache would be overwritten"""
    cached_func = memory.cache(func, *args, **kwargs)
    if not check_cache(cached_func):
        raise CacheOverrideError(cached_func)
    return cached_func


def check_cache(cached_func):
    """checks if cached function is safe to call without overriding cache (adapted from https://github.com/joblib/joblib/blob/7742f5882273889f7aaf1d483a8a1c72a97d57e3/joblib/memory.py#L672)

    Inputs:
        cached_func -- cached function to check

    Returns:
        True if cached function is safe to call, else False

    """

    # Here, we go through some effort to be robust to dynamically
    # changing code and collision. We cannot inspect.getsource
    # because it is not reliable when using IPython's magic "%run".
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
    mmap_mode=None,
    shard_digits=2,
    DB_ATTEMPTS=20,
    wait_exponential = 5,
):

    # Use a base name that includes the full function path and a hash on the function code itself
    base = better_build_func_identifier(func)

    def wrapper(*args, **kwargs):
        """Wrapper for callable to cache arguments and return values."""

        # Get the key for the arguments
        key = hash(
            filter_args(func, ignore, args, kwargs), coerce_mmap=(mmap_mode is not None)
        ).encode("utf-8")

        # Generate a base db location based on the full function path and a hash on the function code itself
        db_loc = ".".join(base)

        # Add a couple of shard digits so that actually we reference a subfolder (string is hex, so two shard digits is 16^2 shards)
        # the numer of shards should be approximately equal to the number of expected simultaneous writers
        if shard_digits > 0:
            db_loc = str(db_loc) + "." + key.decode()[0:shard_digits] + ".sqlite"

        result = NO_CACHE_RESULT

        db_attempts = DB_ATTEMPTS

        time.sleep(np.random.exponential(wait_exponential))

        while db_attempts > 0:
            try:
                with SqliteSingleThreadDict(
                    folder + db_loc,
                ) as cache:

                    #cache.conn._wait_for_initialization()

                    # Grab the cached entry (might be None)
                    if key in cache:
                        try:
                            result = cache[key]
                        except pickle.PicklingError as e:
                            # It's in the cache, but the pickle data is corrupted :(
                            pass

                    # If we need to compute the result, do it, but make sure we only do this once
                    # in case we're hitting it due to cache connection problems.
                    if result == NO_CACHE_RESULT:
                        result = func(*args, **kwargs)
                        cache[key] = result

                    # We're done, don't loop again
                    db_attempts = 0


            except (
                sqlite3.OperationalError,
                AttributeError,
                sqlite3.DatabaseError,
                TimeoutError,
            ) as e:
                # We're probably trying to initialize this DB at the same time as some else
                # Try again a few times
                db_attempts -= 1
                if db_attempts > 0:
                    time.sleep(np.random.exponential(wait_exponential))
                    pass
                else:
                    # We're hit errors too many times, so let's fail
                    raise Exception(
                        f"Error Connecting to {db_loc}"
                    ) from e

        return result

    return wrapper

class SqliteSingleThreadDict(dict):

    def __init__(self, filename=None, tablename='unnamed', journal_mode="TRUNCATE", encode=cloudpickle.dumps,
                 decode=cloudpickle.loads):
        self.filename = filename
        self.journal_mode = journal_mode
        self.encode = encode
        self.decode = decode
        self.tablename = tablename

    def _new_conn(self):
        conn = sqlite3.connect(self.filename, check_same_thread=False, timeout=30)

        with closing(conn.cursor()) as cursor:
            cursor.execute('PRAGMA journal_mode = %s' % self.journal_mode)
            cursor.execute('PRAGMA synchronous=FULL')
            MAKE_TABLE = 'CREATE TABLE IF NOT EXISTS "%s" (key TEXT PRIMARY KEY, value BLOB)' % self.tablename
            cursor.execute(MAKE_TABLE)

        conn.commit()
        return conn

    def __enter__(self):
        if not hasattr(self, 'conn') or self.conn is None:
            self.conn = self._new_conn()
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self, do_log=True, force=False):
        if hasattr(self, 'conn') and self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None

    def __contains__(self, key):
        HAS_ITEM = 'SELECT 1 FROM "%s" WHERE key = ?' % self.tablename
        with closing(self.conn.cursor()) as cursor:
            return len(cursor.execute(HAS_ITEM, (key,)).fetchall()) > 0
        self.conn.commit()

    def __getitem__(self, key):
        with closing(self.conn.cursor()) as cursor:
            GET_ITEM = 'SELECT value FROM "%s" WHERE key = ?' % self.tablename
            item = cursor.execute(GET_ITEM, (key,)).fetchall()
        if len(item) == 0:
            raise KeyError(key)
        value,  = item[0]
        self.conn.commit()
        return self.decode(value)

    def __setitem__(self, key, value):
        with closing(self.conn.cursor()) as cursor:
            ADD_ITEM = 'REPLACE INTO "%s" (key, value) VALUES (?,?)' % self.tablename
            cursor.execute(ADD_ITEM, (key, self.encode(value)))
        self.conn.commit()
