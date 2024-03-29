import operator
import pickle
import uuid

import dask
import numpy as np
from dask.bag import Bag
from dask.bag.core import split
from dask.base import tokenize
from dask.dataframe.io.io import sorted_division_locations
from dask.highlevelgraph import HighLevelGraph
from fsspec.core import open_files
from pympler.asizeof import asizeof


# Register a better method to track the size of complex dictionaries and lists
# (basically pickle and count size). Needed to accurately track data in dask cluster.
@dask.sizeof.sizeof.register(dict)
def sizeof_python_dict(d):
    return asizeof(d)


@dask.sizeof.sizeof.register(list)
def sizeof_python_list(l):
    return asizeof(l)


def _rebalance_ddf(ddf):
    """
    Repartition dask dataframe to ensure that partitions are roughly equal size.
    Assumes `ddf.index` is already sorted.

    Args:
        ddf (dask.core.frame.DataFrame): a dask DataFrame to repartition

    Returns:
        dask.core.frame.DataFrame: a repartitioned version of the input dask DataFrame
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
    """
    Repartition a dask dataframe.

    Args:
        df (dask.core.frame.DataFrame): a dask DataFrame to repartition.
        npartitions (int): a number of partitions to use. Set to -1 to infer.

    Returns:
        dask.core.frame.DataFrame: rebalanced dataframe
    """
    if npartitions == -1:
        npartitions = df.shape[0].compute()
    df = df.repartition(npartitions=npartitions)
    return _rebalance_ddf(df)


def bag_split_individual_partitions(bag):
    """
    Split a bag into as many partitions as possible.

    Args:
        bag (dask.bag.Bag): a dask Bag

    Returns:
        dask.bag.Bag: the input bag, split into more partitions.
    """
    new_name = "repartition-%s" % (tokenize(bag))

    def get_len(partition):
        """
        Get the length of a partition. If the bag is the result of bag.filter(), each
        partition is actually a `filter` object, which has no __len__. In that case,
        convert to a `list` first.

        Args:
            partition (Iterable[dict]): a partition of a dask Bag

        Returns:
            int: The length of the partition; i.e., the number of objects in it.
        """
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


def to_pickles(b, path, name_function=None, compute=True, **kwargs):
    """
    Send a dask dataframe to a series of pickle files.

    Args:
        b (dask.dataframe.core.DataFrame): Dataframe to pickle.
        path (str): Folder location where pickles are written to.
        name_function (function, optional): Function defining how pickle files are
            named. Defaults to None.
        compute (bool, optional): Whether to compute the dataframe before pickling.
            Defaults to True.

    Returns:
        _type_: _description_
    """
    files = open_files(
        path,
        mode="wb",
        encoding=None,
        name_function=name_function,
        num=b.npartitions,
    )

    name = "to-pickle-" + uuid.uuid4().hex
    dsk = {(name, i): (_to_pickle, (b.name, i), f) for i, f in enumerate(files)}
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[b])
    out = type(b)(graph, name, b.npartitions)

    if compute:
        out.compute(**kwargs)
        return [f.path for f in files]
    else:
        return out.to_delayed()


def _to_pickle(data, lazy_file):
    """
    Write data to a pickle file.

    Args:
        data (Any): Object to pickle. Must be pickleable.
        lazy_file (open file-like): file IO stream to write to
    """
    with lazy_file as f:
        pickle.dump(data, f)
