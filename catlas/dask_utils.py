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
    _build_func_identifier,
    extract_first_line,
    JobLibCollisionWarning,
)
from joblib.func_inspect import get_func_name

import os


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
                _build_func_identifier(cached_func.func),
            )
        )


def safe_cache(memory, func, *args, **kwargs):
    cached_func = memory.cache(func, *args, **kwargs)
    if not check_cache(cached_func):
        raise CacheOverrideError(cached_func)
    return cached_func


def check_cache(cached_func):
    """checks if cached function is safe to run without overriding cache (adapted from https://github.com/joblib/joblib/blob/7742f5882273889f7aaf1d483a8a1c72a97d57e3/joblib/memory.py#L672)

    Inputs:
        cached_func -- cached function to check

    Returns:
        True if cached function is safe to run, else False

    """

    # Here, we go through some effort to be robust to dynamically
    # changing code and collision. We cannot inspect.getsource
    # because it is not reliable when using IPython's magic "%run".
    func_code, source_file, first_line = cached_func.func_code_info
    func_id = _build_func_identifier(cached_func.func)

    try:
        old_func_code, old_first_line = extract_first_line(
            cached_func.store_backend.get_cached_func_code([func_id])
        )
    except (IOError, OSError):  # some backend can also raise OSError
        cached_func._write_func_code(func_code, first_line)
        return False
    if old_func_code == func_code:
        return True

    # We have differing code, is this because we are referring to
    # different functions, or because the function we are referring to has
    # changed?

    _, func_name = get_func_name(
        cached_func.func, resolv_alias=False, win_characters=False
    )
    if old_first_line == first_line == -1 or func_name == "<lambda>":
        if not first_line == -1:
            func_description = "{0} ({1}:{2})".format(
                func_name, source_file, first_line
            )
        else:
            func_description = func_name
        warnings.warn(
            JobLibCollisionWarning(
                "Cannot detect name collisions for function '{0}'".format(
                    func_description
                )
            ),
            stacklevel=stacklevel,
        )

    # Fetch the code at the old location and compare it. If it is the
    # same than the code store, we have a collision: the code in the
    # file has not changed, but the name we have is pointing to a new
    # code block.
    if not old_first_line == first_line and source_file is not None:
        possible_collision = False
        if os.path.exists(source_file):
            _, func_name = get_func_name(cached_func.func, resolv_alias=False)
            num_lines = len(func_code.split("\n"))
            with open_py_source(source_file) as f:
                on_disk_func_code = f.readlines()[
                    old_first_line - 1 : old_first_line - 1 + num_lines - 1
                ]
            on_disk_func_code = "".join(on_disk_func_code)
            possible_collision = on_disk_func_code.rstrip() == old_func_code.rstrip()
        else:
            possible_collision = source_file.startswith("<doctest ")
        if possible_collision:
            warnings.warn(
                JobLibCollisionWarning(
                    "Possible name collisions between functions "
                    "'%s' (%s:%i) and '%s' (%s:%i)"
                    % (
                        func_name,
                        source_file,
                        old_first_line,
                        func_name,
                        source_file,
                        first_line,
                    )
                ),
                stacklevel=stacklevel,
            )

    # The function has changed, wipe the cache directory.
    # XXX: Should be using warnings, and giving stacklevel
    if cached_func._verbose > 10:
        _, func_name = get_func_name(cached_func.func, resolv_alias=False)
        cached_func.warn(
            "Function {0} (identified by {1}) has changed"
            ".".format(func_name, func_id)
        )
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
