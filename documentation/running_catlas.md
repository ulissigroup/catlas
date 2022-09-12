# Running Catlas

## Performing an inference run
### Executing a run locally or on a cluster
From the catlas base directory
```
$ python bin/predictions.py configs/path/to/inputs.yml configs/path/to/cluster.py
```

### Executing a run on a slurm managed HPC cluster
We have an example for use on Perlmutter
```
$ sbatch configs/dask_cluster/perlmutter/catlas_run.sh
```
For additional options see [NERSC documentation](https://docs.nersc.gov/jobs/)

We highly recommend trying catlas on your local machine for a trivial test case before trying on slurm.

### Monitoring
You can monitor your run using the dask dashboard. For a local run this should be hosted on port 8787 unless otherwise specified in the terminal.


### Setup using the dask operator on a kubecluster
#### 1. Starting a segementation of the cluster
```
$ kubectl apply -f path/to/cluster_spec.yml
```

An example has been provided in the repo. Its path is `catlas/configs/dask_cluster/dask_operator/catlas-hybrid-cluster.yml` By default, this spins up 2 cpu workers. You can edit the file to change this, or you can scale up and down resources using the commands below.


#### 2. Scaling resources up and down

For cpu workers:


```
$ kubectl scale --replicas=8 daskworkergroup catlas-hybrid-cluster-default-worker-group
```



For gpu workers:

```
$ kubectl scale --replicas=0 daskworkergroup catlas-hybrid-cluster-gpu-worker-group
```

**Note:** `--replicas` sets the *total* number of workers of that type. So if you start with 2 cpu workers and run the first command above, you will have 8. Likewise, if you start with 2 gpu workers and run the second command you will have no gpu workers!



### Shutdown using the dask operator on a kubecluster
At the end of a run you should delete, or scale down your resources. To scale down, see above. To delete:

```
$ kubectl delete daskcluster catlas-hybrid-cluster
```

## Generating parity plots
### Executing a parity run
```
$ python bin/get_parities.py configs/path/to/inputs.yml
```
The input yaml here should be of the same format as one that would be used for an inference run. Adsorbates specified will have specific plots make for each of them. The bulk filters will be applied in the same ways. We recommend not filtering by desired mpid as this can be unnecessarily narrow. For example if you have `filter_by_bulk_ids: ["mp-30"]` try filtering by the following instead:
```
bulk_filters:
  filter_by_num_elements: [1]
  filter_by_element_groups: ["transition metal"]
```
There are other examples of this so be tactful!

## Processing data for parity generation
To make parity plots, we need inference for the same adsorbate-surface configurations which will included in the OC20 validation dataset. Running this inference is currently not supported in catlas because it can easily be done using the OCP infrastructure which has prewritten lmdb files containing the structures of interest. For more information about how to run inference in ocp, see [OCP documentation](https://github.com/Open-Catalyst-Project/ocp/blob/main/TRAIN.md). If the model is an s2ef model, it will output trajectory files in a folder specified in your model config. If the model is an IS2RE direct model it will output inferred energies to an npz file. Catlas has 2 files which each treat one of these cases to process the data so it is usable for parity generation.

### Processing npz files
```
$ python bin/process_npz_for_validation.py --npz_path path/to/npz/file.npz
```

This will make a pickle file of a dataframe which contains all the information that will be needed to make parity plots for the model desired. By default, this will be called whatever the npz was called and be saved to `catlas/catlas/parity/df_pkls/`.


**Advanced**: You may optionally include an additional argument `--dft_df_path path/to/dft/df.pkl` if you would like to use custom dft data. This defaults to the OC20 val set. For more information see Advanced use.

### Processing npz files
```
$ python bin/process_trajectories_for_validation.py --folders path/to/folder1_w_trajs/ path/to/folder2_w_trajs/ --model_id model_name
```

This will make a pickle file of a dataframe which contains all the information that will be needed to make parity plots for the model desired. This will be called by what is specified as `model_name`. It must match the name of the checkpoint for things to work. The output will be saved to `catlas/catlas/parity/df_pkls/`.


**Advanced**: You may optionally include an additional argument `--dft_df_path path/to/dft/df.pkl` if you would like to use custom dft data. This defaults to the OC20 val set. For more information see Advanced Use

## Selecting number of relaxation steps
```
$ python bin/optimize_frame.py path/to/inputs.yml
```

This should be used to select the appropriate number of relaxation steps for S2EF models. It required a preprocessed pkl file containing parity data for that model. It filters that dataframe using the criteria specified in the input yaml file. As with parity, recommend not filtering by desired mpid as this can be unnecessarily narrow. For example if you have `filter_by_bulk_ids: ["mp-30"]` try filtering by the following instead:
```
bulk_filters:
  filter_by_num_elements: [1]
  filter_by_element_groups: ["transition metal"]
```
There are other examples of this so be tactful!


Once filtered, it outputs the frame number which minimizes the MAE for each adsorbate of interest. It also makes plots for the MAE and ME as a function of step number for each adsorbate so a better understanding may be developed. All of this is output into a subdirectory in the outputs folder using the run name specified in the inputs yaml file. 