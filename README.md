Howdy! I am excited that you're interested in parallelizing your predictions.


# Introduction:

The workflow I have made takes as input:
 - list of atomic numbers and element numbers (1 = unary, 2 = binary) or a list of mpids
 - What you have chosen from above as a boolean
 - List of adsorbate smiles
 - Model config yaml
 - Worker spec yaml
 - Number of workers

It generates adslabs for all adsorbates, on all sites, on all surfaces with miller index <= 2. It then uses the model specified in the model config yaml to make direct predictions of the adsorption energy. These values are returned into a dask bag as a list with the following info:
```
return = [mpid,surface_info, adsorbate, energy_now]
```
Here surface info is a tuple of the following form:
```
surface_info = ((m1,m2,m3), shift, top?)
```
Where m1-3 are the miller indices. This is returned to a lazy-loading dask bag which prevents the information from occupying the local RAM. For large datasets, this is necessary. If you are only looking at a small amount of data, you can replace the line 
predictions = predictions_bag.persist()
with 
predictions = predictions_bag.compute()
and the results will be stored in your local RAM and may be queried like a list.


# Getting started:

In addition to the OCP repo, you need to have access to the files in this repo, here is a description of their purpose:
 - predictions.ipynb - the main file! I find a notebook to be more convenient in this case than a .py file because as you will see, you need to perform some downstream operations on your data
 - calculator_upload.py - this instantiates the OCP calculator on each of your dask workers. It also contains a function called predict_E that will be used to make predictions.
 - enumeration_helper_script.py - this has been adapted from Pari’s script with the same name. It contains three functions which are used to generate the slabs, convert the slab object, and then generate the adslab.
 - worker-spec.yaml - as the name implies, this contains the worker specifications.
 - ocpcalc_config.yaml - this contains the checkpoint, model specs, dataset specs, task specs, and optim specs
 - bulk_object_lookup_dict.pkl - this is used to grab the bulk object given the bulk mpid


## File Placement:

calculator_upload.py, predictions.pynb, and enumeration_helper_script.py should be placed in your main ocp folder (where main.py is)


## File Paths:

There are several path references that you have to make sure correctly align with your file structure. I highly recommend using absolute file paths for all paths. This is because the workers use the files too and their default working directory is not home/jovyan, rather it is a subfolder. Here they are:
 - predictions.ipynb - paths are specified as inputs
 - calculator_upload.py - the config path. Make sure it matches the one specified in predictions as well
 - enumeration_helper_script.py - the bulk_db path
 - ocpcalc_config.yml - checkpoint_path, dataset paths (train and val)
I recommend mimicking what I did and place the yamls in the shared-scratch. The workers will be mounting the OC20 data volume and the shared scratch volume the same way you have them mounted to your workspace. Otherwise you will just have to play around with things.


## Volume Mounts:

You have to mount 3 volumes and perform 4 volume mounts to your workers. This is specified in the worker-spec.yml. The oc20data and the scratch-vol volume names should be fine, but make sure the mount path specified mirrors the file structure of your workspace.
Note: oc20data = shared-datasets as specified in the yaml so the OC20 is a subfolder just like it should be in your volume. You have to change the volume mount names for your workspace. Change workspace-ocp to be workspace-”your workspace name here”. The file paths, however, should be the same unless you have a different file structure.


## Docker Image:

You can use my docker image which is in ulissigroup:kubeflow/predictions or make sure you have all the same things. 


## Use:

Once you have things set up, adjust your inputs appropriately. Run cells 1-3 in the notebook. Cell 1 creates the workers and loads the model locally. Cell 2 grabs the adsorbates and bulks from pkl files. Cell 3 sends the work to the dask workers. You can look at the logs of your workers in Rancher. This is useful for troubleshooting problems. You can monitor your progress using the dask dashboard at this url:
https://laikapack-controller.cheme.cmu.edu/k8s/clusters/c-qc7lr/api/v1/namespaces/your_namespace_name/pods/your_pod_name-0:####/proxy/status
Make the appropriate edits (####, your_namespace_name, your_pod_name). The #### is the port number which is printed out in the notebook.


## Downstream use:

If your dataset is small you can simply push it into your local RAM and do with it what you please. If it is not, I would recommend converting the dask bag of results into a dask dataframe. This makes it easy to filter by some characteristic or find the minimum energy on a per surface / per bulk basis. Once you have performed these size reductions, you can convert your dask dataframe into a regular dataframe and take things from there. I have left the cell blocks to perform df operations in the notebook in case you want them.


## Just want adslabs?:
comment out: 
```
# generate prediction mappings
predictions_bag = adslab_bag.map(memory.cache(predict_E))

# execute operations (go to all work)
predictions = predictions_bag.persist() # change to .compute() to push to local RAM
```
and add:
```
adslabs = adslab_bag.persist()
```

Help:

I (Brook) am happy to help! :) 
