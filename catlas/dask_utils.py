from dask.bag.core import split
from dask.dataframe.core import new_dd_object, split_evenly
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from operator import getitem
import numpy as np
import dask.dataframe as dd
from dask.dataframe.io.io import sorted_division_locations


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
