from dask.distributed import Client
from catlas.dask_kube_utils import get_namespace
import dask

# Connect to the cluster
client = Client(
    f"tcp://catlas-hybrid-cluster-service.{get_namespace()}.svc.cluster.local:8786"
)
dask.config.set({"distributed.comm.timeouts.tcp": "240s"})
