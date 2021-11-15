from dask.distributed import Client
from dask_kubernetes import KubeCluster
from dask_kubernetes.objects import make_pod_from_dict, clean_pod_template
from catlas.dask_utils import kube_cluster_new_worker
import dask
from jinja2 import Template

dask.config.set({"distributed.comm.timeouts.connect": 120})
dask.config.set({"distributed.worker.daemon": False})
dask.config.set({"distributed.scheduler.allowed_failures": 20})
dask.config.set({"distributed.comm.retry.count": 20})

with open("configs/debug_configs/scheduler-cpu-dev.yml") as f:
    scheduler_pod_template = make_pod_from_dict(
        dask.config.expand_environment_variables(yaml.safe_load(f))
    )

cluster = KubeCluster(
    pod_template="configs/dask_config/worker-gpu-dev.yml",
    scheduler_pod_template=scheduler_pod_template,
    namespace="zulissi",
    name="dask-catlas-dev",
    scheduler_service_wait_timeout=120,
)
cluster.scale(10)

kube_cluster_new_worker(cluster, "configs/dask_config/workers-cpu-dev.yml")
cluster.scale(60)
client = Client(cluster)
