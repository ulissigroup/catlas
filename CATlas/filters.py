import warnings
import numpy as np
import copy


def bulk_filter(config, dask_df):
    bulk_filters = config["bulk_filters"]

    for name, val in bulk_filters.items():
        if str(val) != "None": # depending on how yaml is created, val may either be "None" or NoneType
            if name == "filter_by_mpids":
                dask_df = dask_df[dask_df.bulk_mpid.isin(val)]
            elif name == "filter_by_acceptable_elements":
                dask_df = dask_df[
                    dask_df.bulk_elements.apply(
                        lambda x, valid_elements: all(
                            [el in valid_elements for el in x]
                        ),
                        valid_elements=val,
                    meta=('bulk_elements', 'bool')
                    )
                ]
            elif name == "filter_by_num_elements":
                dask_df = dask_df[dask_df.bulk_nelements.isin(val)]
            elif name == "filter_by_required_elements":
                dask_df = dask_df[
                    dask_df.bulk_elements.apply(
                        lambda x, required_elements: all(
                            [
                                any([el == req_el for el in x])
                                for req_el in required_elements
                            ]
                        ),
                        required_elements=val,
                    )
                ]
            elif name == "filter_by_object_size":
                dask_df = dask_df[dask_df.bulk_natoms <= val]
            else:
                warnings.warn("Bulk filter is not implemented: " + name)
            if config['output_options']['verbose']:
                print('filter "' + name + '" filtered bulk df down to length ' + str(len(dask_df)))
    return dask_df


def slab_filter(config, dask_df):
    slab_filters = config["slab_filters"]

    for name, val in slab_filters.items():
        if val != "None":
            if name == "filter_by_object_size":
                dask_df = dask_df[dask_df.slab_natoms <= val]
            elif name == "filter_by_max_miller_index":
                dask_df = dask_df[dask_df.slab_max_miller_index <= val]
            else:
                warnings.warn("Slab filter is not implemented: " + name)
    return dask_df


def adsorbate_filter(config, dask_df):
    adsorbate_filters = config["adsorbate_filters"]

    for name, val in adsorbate_filters.items():
        if val != "None":
            if name == "filter_by_smiles":
                dask_df = dask_df[dask_df.adsorbate_smiles.isin(val)]
            else:
                warnings.warn("Adsorbate filter is not implemented: " + name)

    return dask_df
