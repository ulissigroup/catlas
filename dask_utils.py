from dask.bag.core import split
from dask.base import tokenize
import operator
from dask.highlevelgraph import HighLevelGraph
from dask.bag import Bag

def bag_split_individual_partitions(bag):
    new_name = "repartition-%s" % (tokenize(bag))

    def get_len(partition):
        # If the bag is the result of bag.filter(),
        # then each partition is actually a 'filter' object,
        # which has no __len__.
        # In that case, we must convert it to a list first.
        if hasattr(partition, '__len__'):
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
        elif k>1:
            dsk[split_name, i] = (split, (bag.name, i), k)
            for jj in range(k):
                dsk[new_name, j] = (operator.getitem, (split_name, i), jj)
                j += 1

    graph = HighLevelGraph.from_collections(new_name, dsk, dependencies=[bag])
    return Bag(graph, name=new_name, npartitions=sum(nsplits))
