# Making a config
Catlas is highly configurable. There are some examples in `configs/automated_screens` and `configs/tests`. Here we will go through each section of the config and detail the options.

## Memory cache location
This is a required entry and it should be a string of the path to the folder in which results should be cached.


ex.
```memory_cache_location: '/home/jovyan/test_cache'```

## Output options
### Run name
This will be used to make a folder in `outputs` which will house all outputs from your run. Name it something helpful to help future you track down old results.


ex.
```run_name: "OH_on_binary_intermetallics"```

### Pickle final outputs
This boolean specifies whether or not to pickle a final dataframe of your results into the outputs folder. If you are running something so large it can not fit in local memory, you may want to set this to `False`.


ex. 
```pickle_final_output: True```

### Pickle intermediate outputs
This is useful for edge cases. If set to `True`, it will pickle each work partition processed by a worker. This means that many (O(100)-O(1000)) pickle files will populate a subfolder in your outputs folder as they finish up.


ex.
```pickle_intermediate_outputs: False```

### Make parity plots
If `True`, this will perform the same functionality as `bin/get_parities.py` at the beginning of the inference run. If data is available for the model you selected, parity plots will be generated and saved in the outputs folder.

ex.
```make_parity_plots: True```

### Verbose
If `True` this will print the summary dataframe to the terminal at the conclusion of a run. This (as well as pickle final outputs) pushes all results to local memory. If you are running something so large it can not fit in local memory, you may want to set this to `False`.


## Input Options
This is where you specify the file paths to the bulk and adsorbate input files. This is required.

ex.
```
input_options:
  bulk_file: 'catlas/bulk_structures/ocp_bulks_w_properties.db'
  adsorbate_file: 'catlas/adsorbate_structures/adsorbates.pkl'
```

## Bulk Filters
This is where you specify how you would like to downselect the material design space. There are several options:
1. Bulk ids to include: a list of bulk ids to include
```filter_by_bulk_ids: ['mp-126','mp-30', 'mp-81', 'mp-13', 'mp-79']```
2. Acceptable elements: a list of element symbols to include. If you say "Au" and "Ag", and materials containing only Au and Ag will be included.
```filter_by_acceptable_elements: ["Au", "Ag"]```
3. Bulk ids to ignore: a list of bulk ids to exclude
```filter_ignore_bulk_ids: ['mp-126','mp-30', 'mp-81', 'mp-13', 'mp-79']```
4. Number of elements: a list of the numbers which are acceptable i.e. `[2, 3]` gives you binary and ternary materials
```filter_by_num_elements: [2, 3]```
5. Required elements: a list of elements which must be in each material. If you say "Cu", then all copper alloys will be considered.
```filter_by_required_elements: ["Cu"]```
6. Number of atoms: this is useful to avoid very large structures which are costly to compute. It filters out all unit cell structures with more atoms than the number specified.
```filter_by_object_size: 60```
7. Active-host paradigm: allows you to find materials containing elements you are interested in where at least one element is coming from one list of materials and the other element coming from another list. The example here would give Zinc alloys with Pd, Ag, and/ or Cu.
```
filter_by_elements_active_host:
  active: ["Pd", "Ag", "Cu"]
  host: ["Zn"]
```
8. Filter by element groups: A list of periodic table groups which are of interest. The groups should be specified as a list of any of the following: `["transition metal", "post-transition metal", "metalloid", "rare earth metal", "alkali", "alkaline", "alkali earth", "chalcogen", "halogen"]`
```filter_by_element_groups: ["transition metal"]```
9. Energy above hull: a float of the maximum energy above hull you would like to consider
```filter_by_bulk_e_above_hull: 0.05```
10. Band gap: This requires one or both of the minimum and maximum band gap you would like to be considered to be specified. If both are, then materials in the range will be filtered for. If only the maximum is, then anything with a band gap less than the value given will be filtered for. If only the minimum is, then anything with a band gap greater than the value given will be filtered for.
```
filter_by_bulk_band_gap:
  min_gap: 0.1
  max_gap:0.3
```
11. Pourbaix stability: This is only supported for Materials Project materials as of now. It selects the Pourbaix stable materials under the conditions you specify. Conditions may be specified as a range or as a list of specific values. You may also specify a maximum decoposition energy.  Step size is the increment that will be used for the interval. If the material is stable at any of the conditions considered, it will not be filtered out. A path to an lmdb file containing Pourbaix info for your materials is required. If the file path doesnt exist, the Pourbaix info will be queried from MP as a part of the run. This does take a bit of time. Be careful of having to many workers querying to avoid being black listed.
```
filter_by_pourbaix_stability: 
  max_decomposition_energy: 0.05
  lmdb_path: "catlas/pourbaix_diagrams/20220222_query.lmdb"
  V_lower: -0.5
  V_upper: 1
  V_step: 0.05
  pH_lower: 0
  pH_upper: 2
```
OR
```
  filter_by_pourbaix_stability: 
    max_decomposition_energy: 0.5
    lmdb_path: "catlas/pourbaix_diagrams/20220222_query.lmdb"
    conditions_list:
      - pH: 1
        V: -1.2
      - pH: 14
        V: 1.2
```
12. Randomly sample: specify the fraction of materials you would like to randomly select. The example here randomly selects 5% of materials.
```filter_fraction: 0.05```

## Adsorbate filters
There is only one option for adsorbate filters which is to filter by their SMILES string. Provide a list of the SMILES strings for adsorbates you would like to consider. An exhaustive list of those in OC20 is included here in our example.


ex.
```
adsorbate_filters:
    filter_by_smiles: ['*C', '*C*C', '*CCH', '*CCH2', '*CCH2OH', '*CCH3', '*CCHO', '*CCHOH', '*CCO', '*CH', '*CH*CH', '*CH*COH', '*CH2', '*CH2*O', '*CH2CH2OH', '*CH2CH3', '*CH2OH', '*CH3', '*CH4', '*CHCH2', '*CHCH2OH', '*CHCHO', '*CHCHOH', '*CHCO', '*CHO', '*CHO*CHO', '*CHOCH2OH', '*CHOCHOH', '*CHOH', '*CHOHCH2', '*CHOHCH2OH', '*CHOHCH3', '*CHOHCHOH', '*CN', '*COCH2O', '*COCH2OH', '*COCH3', '*COCHO', '*COH', '*COHCH2OH', '*COHCH3', '*COHCHO', '*COHCHOH', '*COHCOH', '*H', '*N', '*N*NH', '*N*NO', '*N2', '*NH', '*NH2', '*NH2N(CH3)2', '*NH3', '*NHNH', '*NO', '*NO2', '*NO2NO2', '*NO3', '*NONH', '*O', '*OCH2CH3','*OCH2CHOH', '*OCH3', '*OCHCH3', '*OH', '*OH2', '*OHCH2CH3', '*OHCH3', '*OHNH2', '*OHNNCH3', '*ONH', '*ONN(CH3)2', '*ONNH2', '*ONOH', 'CH2*CO', '*CO', '*CH2*CH2', '*COHCH2, '*NHN2', '*NNCH3', '*OCHCH2', '*ONNO2']
```

## Slab Filters
Downselect slabs before performing adsorbate placement and inference.
1. Object size: filters out any slabs with more atoms than the number specified here. This is useful to avoid need many calculations for very large surfaces.
```filter_by_object_size: 100```
2. Maximum miller index: will filter out any slab where the any miller index excedes the value specified. i.e. 2 will filter out (-2,1,0) and (2,1,1)
```filter_by_max_miller_index: 1```

## Adslab Prediction Steps
Runs may be set up so that inference is performed sequentially. The idea here is that you may want to use a cheap, less accurate model to downselect first, and then perform more expensive, accurate inference. If you would not like to do this, simply use one step instead.


ex.
```
adslab_prediction_steps:
  - step_type: inference
    gpu: true
    batch_size: 8
    label: 'dE_gemnet_is2re_finetuned'
    checkpoint_path: 'ocp_checkpoints/private_checkpoints/gemnet-is2re-finetuned-11-01.pt'
```
OR
```
adslab_prediction_steps:
  - step_type: inference
    gpu: true
    batch_size: 8
    label: 'dE_gemnet_is2re_finetuned'
    checkpoint_path: 'ocp_checkpoints/private_checkpoints/gemnet-is2re-finetuned-11-01.pt'
  - step_type: filter_by_adsorption_energy_target
    adsorbate_smiles: '*CO'
    target_value: -0.6
    range_value: 0.2
    filter_column: 'min_dE_gemnet_is2re_finetuned'
  - step_type: inference
    gpu: true
    batch_size: 8
    label: 'dE_gemnet_oc_large_s2ef_all_md'
    checkpoint_path: 'ocp_checkpoints/public_checkpoints/gemnet_oc_large_s2ef_all_md.pt'
```
OR
```
adslab_prediction_steps:
  - step_type: inference
    gpu: true
    batch_size: 8
    label: 'dE_gemnet_is2re_finetuned'
    checkpoint_path: 'ocp_checkpoints/private_checkpoints/gemnet-is2re-finetuned-11-01.pt'
  - step_type: filter_by_adsorption_energy_target
    adsorbate_smiles: '*CO'
    min_value: -0.8
    max_value: -0.4
    filter_column: 'min_dE_gemnet_is2re_finetuned'
  - step_type: inference
    gpu: true
    batch_size: 8
    label: 'dE_gemnet_oc_large_s2ef_all_md'
    checkpoint_path: 'ocp_checkpoints/public_checkpoints/gemnet_oc_large_s2ef_all_md.pt'
    number_steps: 98
```

1. `step_type`: this should be `inference`, `filter_by_adsorption_energy`, or `filter_by_adsorption_energy_target`. As their names imply `inference` is an inference step, `filter_by_adsorption_energy_target` filters surfaces that are within a range near your target value, and `filter_by_adsorption_energy` filters surfaces by whether a previous inference step predicts within a range of values you specify. 
2. `gpu`: (inference step only) A boolean for if an inference step should use gpus
3. `batch_size`: (inference step only) The number of adslab configs which will be considered in an inference batch. If you have very large objects you may have to decrease this so they fit in memory.
4. `label`: (inference step only) What the inference step will be named in the output dataframe. Should be the name of the model or something useful to you
5. `checkpoint_path`: (inference step only) file path to the desired pretrained model checkpoint file
6. `number_steps`: (relaxation inference step only) The number of relaxation steps to take. If unspecified, this defaults to 200.
7. `adsorbate_smiles`: (filter step only) SMILES string of the adsorbate to filter on
8. `filter_column`: (filter step only) the df column to which the filter should be applied. For most use cases, this should be "min_" + the previous inference step's label
9. `min_value`: (`filter_by_adsorption_energy` only) The minimum value in the desired range. If unspecified, this defaults to -infinity.
10. `max_value`: (`filter_by_adsorption_energy` only) The maximum value in the desired range. If unspecified, this defaults to +infinity.
11. `target_value`: (`filter_by_adsorption_energy_target` only) The target value.
12. `range_value`: (`filter_by_adsorption_energy_target` only) The range to consider (i.e. `target_value - range_value` -> `target_value + range_value will be used`) If a range value is not specified, then 0.5 is used by default.