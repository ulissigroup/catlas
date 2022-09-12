# Prediction runs (predictions.py)
## Dataframe of results (results_df.pkl)
All information from a run can be found in the `results_df.pkl` file. There is information that memorializes the conditions of the run and the results. Here is a samle of the columns in the dataframe, which have been appropriately named.
```
['bulk_id', 'bulk_data_source', 'bulk_natoms', 'bulk_xc',
'bulk_nelements', 'bulk_elements', 'bulk_e_above_hull', 'bulk_band_gap',
'bulk_structure', 'slab_surface_object', 'surface_structure',
'slab_millers', 'slab_max_miller_index', 'slab_shift', 'slab_top',
'slab_natoms', 'nuclearity_info', 'adsorbate_atoms', 'adsorbate_smiles',
'adsorbate_bond_indices', 'adsorbate_data_source',
'dE_{model_id}', 'min_dE_{model_id}',
'atoms_min_dE_{model_id}_initial', 'adslab_atoms']
```


## Sankey Diagram
The Sankey diagram allows users to easily visualize the enumeration and filtering steps to see how many objects are enumerated and how many are filtered out. For this simple example where 6 unary materials were considered, over 1200 inference calculations were performed!
![sankey example](outputs/sankey.png)


## Parity Plots
See Parity Plot Generation 

## Itermediate pickles
These are pickle files of each of the dask paritions as they finish up. They are generated in a folder named `intermediate_pkls` and individually named the number of the partition.

# Step Number Optimization (optimize_frame.py)
## Summary of results (summary.pkl)
A dataframe memorializing the optimization: including the MAE, optimal frame number, and the per frame MAE and ME.
![frame summary](outputs/summary_frame_opt.png)


## Plot of ME and MAE v. frame number
To provide more inside, the ME and MAE is plotted v. frame number.

![mae me plot](outputs/mae_v_frame.pdf)


# Parity Plot Generation (get_parities.py)
A plot file is made for each of the adsorbates specified in the input yaml file. One additional plot is made which is general for all adsorbats. Some examples are shown below:

![H only](outputs/H_parity.pdf)

![general](outputs/general_parity.pdf)

1. **overall** = all data splits 
2. **id** = in domain
3. **ood_ads** = out of domain adsorbate (i.e. the adsorbate did not appear in the training set)
4. **ood_cat** = out of domain material (i.e. systems derived from that specific bulk structure did not appear in the training set)
4. **ood_both** = both the adsorbate and material were out of domain

The number of datapoints displayed and the mean absolute error (MAE) for that data subset are also shown. A parity line is drawn for ease of use, and a linear regression is performed and plotted to further and understanding of deviance from parity.

