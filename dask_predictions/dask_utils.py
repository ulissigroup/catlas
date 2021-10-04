from dask.bag.core import split
from dask.dataframe.core import new_dd_object, split_evenly
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from operator import getitem


def dataframe_split_individual_partitions(df):
    new_name = "repartition-%s" % (tokenize(df))

    df = df.persist()
    nsplits = df.map_partitions(len).compute()

    dsk = {}
    split_name = "split-{}".format(tokenize(df, nsplits))
    j = 0
    for i, k in enumerate(nsplits):
        if k == 1:
            dsk[new_name, j] = (df._name, i)
            j += 1
        elif k > 1:
            dsk[split_name, i] = (split_evenly, (df._name, i), k)
            for jj in range(k):
                dsk[new_name, j] = (getitem, (split_name, i), jj)
                j += 1

    divisions = [None] * (1 + sum(nsplits))
    graph = HighLevelGraph.from_collections(new_name, dsk, dependencies=[df])
    return new_dd_object(graph, new_name, df._meta, divisions)
