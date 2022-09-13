# catlas
![catlas overview](https://github.com/ulissigroup/catlas/blob/main/documentation/catlas_logo.png?raw=True)

## Repo graphical overview
![catlas overview](https://github.com/ulissigroup/catlas/blob/main/documentation/catlas_overview.png?raw=true)

## Detailed documentation
[https://ulissigroup.cheme.cmu.edu/catlas/intro.html](https://ulissigroup.cheme.cmu.edu/catlas/intro.html)

## Installation
 - Install ocp:
  - `$ git clone git@github.com: Open-Catalyst-Project/ocp`
  - `$ cd ~/ocp && python setup.py develop`
 - Install catlas:
  - `$ git clone git@github.com: ulissigroup/catlas`
  - `$ cd ~/catlas && python setup.py develop`

## Large file handling
Add large files to their appropriate place in the repo
 - [Model checkpoints](https://github.com/Open-Catalyst-Project/ocp/blob/main/MODELS.md) -> catlas/ocp_checkpoints
 - Inference on validation data for parity (link to download coming soon!)

## Usage
### Local / kubernetes cluster
`$ python bin/predictions.py configs/path/to/config.yml configs/path/to/cluster.py`

more info about using dask operator to manage resources on a kubectl cluster: [kubernetes.dask.org](https://kubernetes.dask.org/en/latest/operator.html)

### HPC (Perlmutter)
`$ sbatch configs/dask_cluster/perlmutter/catlas_run.sh`

For additional options see [NERSC documentation](https://docs.nersc.gov/jobs/)

## Monitoring
The default dashboard port is 8787
- localhost:8787/status
