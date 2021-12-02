from dask.bag.core import split
from dask.dataframe.core import new_dd_object, split_evenly
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
import numpy as np
import dask.dataframe as dd
from dask.dataframe.io.io import sorted_division_locations
import operator
from dask.bag import Bag
import pickle
from dask_kubernetes.objects import make_pod_from_dict, clean_pod_template
import yaml
import dask
import joblib.memory
from joblib.memory import (
    extract_first_line,
    JobLibCollisionWarning,
)
from joblib.func_inspect import get_func_name
from tokenize import open as open_py_source
import os


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
    return os.path.join(*parts)


joblib.memory._build_func_identifier = better_build_func_identifier


def hash_func(func):
    """Hash the function id, its file location, and the function code"""
    func_code_h = hash(getattr(func, "__code__", None))
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


def _rebalance_ddf(ddf):
    """Repartition dask dataframe to ensure that partitions are roughly equal size.

    Assumes `ddf.index` is already sorted.
    """
    if not ddf.known_divisions:  # e.g. for read_parquet(..., infer_divisions=False)
        # ddf = ddf.reset_index().set_index(ddf.index.name, sorted=True)
        ddf["x"] = 1
        ddf["x"] = ddf.x.cumsum()
        ddf = ddf.set_index("x", sorted=True)
    index_counts = ddf.map_partitions(
        lambda _df: _df.index.value_counts().sort_index()
    ).compute()
    index = np.repeat(index_counts.index, index_counts.values)
    divisions, _ = sorted_division_locations(index, npartitions=ddf.npartitions)
    return ddf.repartition(divisions=divisions)


def split_balance_df_partitions(df, npartitions):
    if npartitions == -1:
        npartitions = df.shape[0].compute()
    df = df.repartition(npartitions=npartitions)
    return _rebalance_ddf(df)


def bag_split_individual_partitions(bag):
    new_name = "repartition-%s" % (tokenize(bag))

    def get_len(partition):
        # If the bag is the result of bag.filter(),
        # then each partition is actually a 'filter' object,
        # which has no __len__.
        # In that case, we must convert it to a list first.
        if hasattr(partition, "__len__"):
            return len(partition)
        return len(list(partition))

    bag = bag.persist()
    nsplits = bag.map_partitions(get_len).compute()

    dsk = {}
    split_name = "split-{}".format(tokenize(bag, nsplits))
    j = 0
    for i, k in enumerate(nsplits):
        if k == 1:
            dsk[new_name, j] = (bag.name, i)
            j += 1
        elif k > 1:
            dsk[split_name, i] = (split, (bag.name, i), k)
            for jj in range(k):
                dsk[new_name, j] = (operator.getitem, (split_name, i), jj)
                j += 1

    graph = HighLevelGraph.from_collections(new_name, dsk, dependencies=[bag])
    return Bag(graph, name=new_name, npartitions=sum(nsplits))


class SizeDict(dict):
    def __sizeof__(self):
        return len(pickle.dumps(self))


def kube_cluster_new_worker(cluster, config_path):
    with open(config_path) as f:
        worker_pod_template = make_pod_from_dict(
            dask.config.expand_environment_variables(yaml.safe_load(f))
        )
        clean_worker_template = clean_pod_template(
            worker_pod_template, pod_type="worker"
        )
        cluster.pod_template = cluster._fill_pod_templates(
            clean_worker_template, pod_type="worker"
        )
        cluster.new_spec["options"]["pod_template"] = cluster.pod_template


def check_if_memorized(input, memorized_func, *args, **kwargs):
    memorized = memorized_func.check_call_in_cache(input, *args, **kwargs)
    if memorized:
        return None
    else:
        return input


def cache_if_not_cached(input, memorized_func, *args, **kwargs):
    if input != None:
        memorized_func.call(input, *args, **kwargs)
    return None


def load_cache(input, memorized_func, memorized_cache, *args, **kwargs):
    return memorized_func(input, *args, **kwargs)
