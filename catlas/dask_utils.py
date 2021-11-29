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
import subprocess

def get_namespace():
    ns_str = subprocess.run('kubectl describe sa default | grep Namespace', shell=True, stdout=subprocess.PIPE, check=True).stdout.decode('utf-8')
    ns = ns_str.split(' ')[-1].replace('\n','')
    return ns

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
