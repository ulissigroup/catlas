# catlas
![catlas overview](https://github.com/ulissigroup/catlas/blob/main/catlas_overview.png?raw=true)
## Installation
 - Install ocp:
  - `$ git clone git@github.com: Open-Catalyst-Project/ocp`
  - `$ cd ~/ocp && python setup.py develop`
 - Install catlas:
  - `$ git clone git@github.com: ulissigroup/catlas`
  - `$ cd ~/catlas && python setup.py develop`

## Large file handling
 ### Ulissigroup internal: Create symbolic links for important large files
 `ln -s /home/jovyan/shared-scratch/catlas/ocp_checkpoints`
 `ln -s /home/jovyan/shared-scratch/catlas/npz-files .catlas/parity/`
 `ln -s /home/jovyan/shared-scratch/catlas/pourbaix_diagrams ./catlas/`
 
 ### Other: Add large files to their appropriate place in the repo
 [Model checkpoints](https://github.com/Open-Catalyst-Project/ocp/blob/main/MODELS.md) -> catlas/ocp_checkpoints
 Inference on validation data for parity (link to download coming soon!)

## Usage
### Local / kubernetes cluster
`$ python bin/predictions.py configs/path/to/config.yml configs/path/to/cluster.py`

### HPC (Perlmutter)
`$ sbatch configs/dask_cluster/perlmutter/catlas_run.sh`
For additional options see [NERSC documentation](https://docs.nersc.gov/jobs/)

## Monitoring
### Ulissigroup internal:
- https://laikapack-controller.cheme.cmu.edu/k8s/clusters/c-qc7lr/api/v1/namespaces/$namespace/pods/$podname-0:8787/proxy/status (local cluster)
- https://laikapack-controller.cheme.cmu.edu/k8s/clusters/c-qc7lr/api/v1/namespaces/$namespace/services/dask-catlas-dev:8787/proxy/status (scheduler)

At the end of a run, delete extra pods and services:
- `kubectl delete service -l 'app=dask'`
- `kubectl delete pod -l 'app=dask'`
- `kubectl delete poddisruptionbudgets dask-catlas-dev`

### Other:
- localhost:8787/status





