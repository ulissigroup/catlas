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
from pathlib import Path


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


open_lmdbs = {}


def lmdb_memoize(folder, func, ignore=(), mmap_mode=None, shard_digits=2):
    base = better_build_func_identifier(func)

    def wrapper(*args, **kwargs):
        """Wrapper for callable to cache arguments and return values."""
        key = hash(
            filter_args(func, ignore, args, kwargs), coerce_mmap=(mmap_mode is not None)
        ).encode("utf-8")

        lmdb_loc = folder + ".".join(base)
        if shard_digits > 0:
            lmdb_loc = str(lmdb_loc) + "/" + key.decode()[0:shard_digits] + "/"

        if lmdb_loc in open_lmdbs and open_lmdbs[lmdb_loc] != 'creating':
            lmdb_env = open_lmdbs[lmdb_loc]
        elif lmdb_loc in open_lmdbs and open_lmdbs[lmdb_loc] == 'creating':
            connected = False
            while open_lmdbs[lmdb_loc] == 'creating':
                pass
            lmdb_env = open_lmdbs[lmdb_loc]
        else:
            open_lmdbs[lmdb_loc] = 'creating'

            Path(lmdb_loc).mkdir(parents=True, exist_ok=True)

            connected = False

            while not connected:
                try:
                    lmdb_env = lmdb.Environment(
                        lmdb_loc,
                        map_size=1024 * 1024 * 1024 * 1024,
                        metasync=True,
                        meminit=False,
                        lock=True,
                        max_dbs=0,
                        max_readers=1000,
                    )
                    connected = True
                except lmdb.LockError:
                    pass

            open_lmdbs[lmdb_loc] = lmdb_env

        calculation_required = False

        retrieved = False
        while not retrieved:
            try:
                with lmdb_env.begin(write=False) as txn:
                    result = txn.get(key)
                retrieved = True
            except lmdb.LockError:
                print("Lock Error! Trying Again!")
                pass

        if result is None:
            calculation_required = True

        if calculation_required:
            computed_result = func(*args, **kwargs)

            committed = False

            while not committed:
                try:
                    with lmdb_env.begin(write=True) as txn:
                        result = txn.get(key)

                        if result is None:
                            txn.put(key, cloudpickle.dumps(computed_result))
                            result = computed_result
                        else:
                            result = cloudpickle.loads(result)

                        committed = True
                except lmdb.LockError:
                    print("Lock Error! Trying Again!")
                    pass
        else:
            result = cloudpickle.loads(result)

        return result

    return wrapper
