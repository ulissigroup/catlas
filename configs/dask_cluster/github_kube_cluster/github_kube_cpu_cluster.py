from dask.distributed import Client
from dask_kubernetes import KubeCluster
import os
from jinja2 import Template
from dask_kubernetes.common.objects import make_pod_from_dict

dask.config.set({"distributed.comm.timeouts.connect": 240})

# Complet the worker template
template = Template(
    open("configs/dask_cluster/github_kube_cluster/worker-cpu-github.tmpl").read()
)
with open(
    "./configs/dask_cluster/github_kube_cluster/worker-cpu-github.yml", "w"
) as fhandle:
    fhandle.write(template.render(**os.environ))

# Load the scheduler template
with open("configs/dask_cluster/github_kube_cluster/scheduler.yml") as f:
    scheduler_pod_template = make_pod_from_dict(
        dask.config.expand_environment_variables(yaml.safe_load(f))
    )

# Start the cluster and scale
cluster = KubeCluster(
    pod_template="configs/dask_cluster/github_kube_cluster/worker-cpu-github.yml",
    scheduler_pod_template=scheduler_pod_template,
    namespace="zulissi",
    name=f"""dask-catlas-{os.environ["GITHUB_RUN_ID"]}""",
    scheduler_service_wait_timeout=480,
)
cluster.scale(2)

# Connect to the cluster
client = Client(cluster)
