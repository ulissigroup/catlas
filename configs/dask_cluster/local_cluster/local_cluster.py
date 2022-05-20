from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=4, threads_per_worker=1)
client = Client(cluster)
