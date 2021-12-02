# catlas

## Installation
 - Install git-lfs:
  - `$ pip install git-lfs`
  - `$ git-lfs install`
 - Install ocp:
  - `$ git clone git@github.com: Open-Catalyst-Project/ocp`
  - `$ cd ~/ocp && python setup.py develop`
 - Install catlas:
  - `$ git clone git@github.com: ulissigroup/catlas`
  - `$ cd ~/catlas && python setup.py develop`

## Usage

`$ python bin/predictions.py configs/path/to/config.yml configs/path/to/cluster.py`

## Automated screens:

This repo runs automated screens either on pushing to the repo or locally.
Find scheduler pods using `kubectl get po -l 'dask.org/component=scheduler'`

Monitor a pod at the url
https://laikapack-controller.cheme.cmu.edu/k8s/clusters/c-qc7lr/api/v1/namespaces/$namespace/pods/$podname-0:8787/proxy/status (local cluster)
https://laikapack-controller.cheme.cmu.edu/k8s/clusters/c-qc7lr/api/v1/namespaces/$namespace/services/dask-catlas-dev:8787/proxy/status (scheduler)

At the end of a run, delete extra pods and services:
- `kubectl delete service -l 'app=dask'`
- `kubectl delete po -l 'app=dask'`
- `kubectl delete poddisruptionbudgets dask-catlas-dev`
