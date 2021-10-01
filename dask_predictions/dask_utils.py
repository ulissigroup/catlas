from dask.bag.core import split
from dask.dataframe.core import repartition

def dataframe_split_individual_partitions(df):
    df = df.persist()
    return df.repartition(list(range(df.shape[0])))

