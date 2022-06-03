from dask.distributed import Client
from catlas.dask_kube_utils import get_namespace


# Connect to the cluster
client = Client(f'tcp://catlas-hybrid-cluster-service.{get_namespace()}.svc.cluster.local:8786')
