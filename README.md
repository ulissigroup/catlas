# dask-predictions
files for dask parallelization of OCPCalculator energy predictions. Assumes you have the ocp repo cloned, access to laikapack, and access to the shared volume mounts on laikapack

Usage:
- make a volume workspace-inference
- start a notebook server mounting workspace-inference and normal shared-datasets/shared-scratch
- sudo apt update && sudo apt install git-lfs
- git clone ocp, checkout the ocpcalc_fix branch, python setup.py develop
- git clone dask-predictions, python setup.py develop
- go to dask-predictions, then python bin/predictions.py config/inputs_nitrides_kube.yml. I set one up just to see if I could get the nitride thing to run, but don't plan on actually seeing that through
