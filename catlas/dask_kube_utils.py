import yaml
import dask
from dask_kubernetes.common.objects import make_pod_from_dict, clean_pod_template
import subprocess


def kube_cluster_new_worker(cluster, config_path):
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
    ns_str = subprocess.run(
        "kubectl describe sa default | grep Namespace",
        shell=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.decode("utf-8")
    ns = ns_str.split(" ")[-1].replace("\n", "")
    return ns
