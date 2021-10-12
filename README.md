# catlas
files for dask parallelization of OCPCalculator energy predictions. Assumes you have the ocp repo cloned, access to laikapack, and access to the shared volume mounts on laikapack

Before running inference, run the following commands:
 - `cd ~/ocp && python setup.py develop`
 - `cd ~/CATlas && python setup.py develop`

To run:
`python -i bin/predictions.py config/something.yml`

How to use kubeflow workers for predictions:
- make a volume workspace-inference
- start a notebook server mounting workspace-inference and normal shared-datasets/shared-scratch
- sudo apt update && sudo apt install git-lfs
- git clone ocp, checkout the ocpcalc_fix branch, python setup.py develop
- git clone CATlas, python setup.py develop
- go to CATlas, then python bin/predictions.py config/inputs_nitrides_kube.yml. I set one up just to see if I could get the nitride thing to run, but don't plan on actually seeing that through


## Automated screens:

This repo runs automated screens either on pushing to the repo or locally.
Find scheduler pods using `kubectl get po -l 'dask.org/component=scheduler'`

Monitor a pod at the url https://laikapack-controller.cheme.cmu.edu/k8s/clusters/c-qc7lr/api/v1/namespaces/\<namespace>/pods/\<podname>:8787/proxy/status

At the end of a run, delete extra pods and services:
 
- `kubectl delete service -l 'app=dask'`
- `kubectl delete po -l 'app=dask'`
- `kubectl delete poddisruptionbudgets dask-catlas-dev`
