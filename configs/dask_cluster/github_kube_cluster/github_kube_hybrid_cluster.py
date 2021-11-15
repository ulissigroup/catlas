from dask.distributed import Client
from dask_kubernetes import KubeCluster
from dask_kubernetes.objects import make_pod_from_dict, clean_pod_template
from catlas.dask_utils import kube_cluster_new_worker
import dask
import os
from jinja2 import Template

dask.config.set({"distributed.comm.timeouts.connect": 240})
dask.config.set({"distributed.scheduler.allowed_failures": 20})
dask.config.set({"distributed.comm.retry.count": 20})

# Get the template for the scheduler
with open("configs/dask_cluster/github_kube_cluster/scheduler.yml") as f:
    scheduler_pod_template = make_pod_from_dict(
        dask.config.expand_environment_variables(yaml.safe_load(f))
    )


# Complete the template for GPU workers
template = Template(
    open("configs/dask_cluster/github_kube_cluster/worker-gpu-github.tmpl").read()
)
with open(
    "configs/dask_cluster/github_kube_cluster/worker-gpu-github.yml", "w"
) as fhandle:
    fhandle.write(template.render(**os.environ))


# Start the dask cluster with some gpu workers
cluster = KubeCluster(
    pod_template="configs/dask_cluster/github_kube_cluster/worker-gpu-github.yml",
    scheduler_pod_template=scheduler_pod_template,
    namespace="zulissi",
    name="dask-catlas-{{ GITHUB_RUN_ID }}",
    scheduler_service_wait_timeout=480,
)
cluster.scale(4)

# Switch to CPU workers and scale it further
template = Template(
    open("configs/dask_cluster/github_kube_cluster/worker-cpu-github.tmpl").read()
)
with open(
    "./configs/dask_cluster/github_kube_cluster/worker-cpu-github.yml", "w"
) as fhandle:
    fhandle.write(template.render(**os.environ))

kube_cluster_new_worker(
    cluster, "configs/dask_cluster/github_kube_cluster/worker-cpu-github.yml"
)
cluster.scale(60)

# Connect to the cluster
client = Client(cluster)
