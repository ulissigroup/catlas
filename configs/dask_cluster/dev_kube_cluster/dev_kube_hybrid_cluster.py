from dask.distributed import Client
from dask_kubernetes import KubeCluster
from dask_kubernetes.objects import make_pod_from_dict, clean_pod_template
from catlas.dask_kube_utils import kube_cluster_new_worker, get_namespace
import dask
import os
from jinja2 import Template

dask.config.set({"distributed.comm.timeouts.connect": 240})
dask.config.set({"distributed.scheduler.allowed_failures": 20})
dask.config.set({"distributed.comm.retry.count": 20})

# Get the template for the scheduler
with open("configs/dask_cluster/dev_kube_cluster/scheduler.yml") as f:
    scheduler_pod_template = make_pod_from_dict(
        dask.config.expand_environment_variables(yaml.safe_load(f))
    )


# Complete the template for GPU workers
template = Template(
    open("configs/dask_cluster/dev_kube_cluster/worker-gpu.tmpl").read()
)
with open("configs/dask_cluster/dev_kube_cluster/worker-gpu.yml", "w") as fhandle:
    fhandle.write(template.render(**os.environ))

# Start the dask cluster with some gpu workers
cluster = KubeCluster(
    pod_template="configs/dask_cluster/dev_kube_cluster/worker-gpu.yml",
    scheduler_pod_template=scheduler_pod_template,
    namespace=get_namespace(),
    name="dask-catlas-dev",
    scheduler_service_wait_timeout=240,
)
cluster.scale(4)

# Switch to CPU workers and scale it further
template = Template(
    open("configs/dask_cluster/dev_kube_cluster/worker-cpu.tmpl").read()
)
with open("./configs/dask_cluster/dev_kube_cluster/worker-cpu.yml", "w") as fhandle:
    fhandle.write(template.render(**os.environ))

kube_cluster_new_worker(cluster, "configs/dask_cluster/dev_kube_cluster/worker-cpu.yml")
# cluster.scale(80)

# Connect to the cluster
client = Client(cluster)
