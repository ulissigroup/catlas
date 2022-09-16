import subprocess

import dask
import yaml
from dask_kubernetes.common.objects import (
    clean_pod_template,
    make_pod_from_dict,
)


def kube_cluster_new_worker(cluster, config_path):
    """
    Generate a new kubernetes worker.

    Args:
        cluster (dask.distributed.client.Client): a dask cluster to run code on.
        config_path (str): a file path containing a config yml file defining the
            specifications of the new worker
    """
    with open(config_path) as f:
        worker_pod_template = make_pod_from_dict(
            dask.config.expand_environment_variables(yaml.safe_load(f))
        )
        clean_worker_template = clean_pod_template(
            worker_pod_template, pod_type="worker"
        )
        cluster.pod_template = cluster._fill_pod_templates(
            clean_worker_template, pod_type="worker"
        )
        cluster.new_spec["options"]["pod_template"] = cluster.pod_template


def get_namespace():
    """
    Return the Kubernetes namespace this code is run in.

    Returns:
        str: the Kubernetes namespace this code is running in.
    """
    ns_str = subprocess.run(
        "kubectl describe sa default | grep Namespace",
        shell=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.decode("utf-8")
    ns = ns_str.split(" ")[-1].replace("\n", "")
    return ns
