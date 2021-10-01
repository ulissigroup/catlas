import warnings
import numpy as np


def bulk_filter(config, dask_df):
    bulk_filters = config["bulk_filters"]

    for name, val in bulk_filters.items():
        if val != "None":
            if name == "filter_by_mpids":
                dask_df = dask_df[dask_df.bulk_mpid.isin(val)]
            elif name == "filter_by_Pourbaix_stability":
                pass
            elif name == "filter_by_elements":
                dask_df = dask_df[
                    dask_df.bulk_elements.apply(lambda x: all([el in val for el in x]))
                ]
            elif name == "filter_by_num_elements":
                dask_df = dask_df[dask_df.bulk_elements.apply(len).isin(val)]
            elif name == "filter_by_object_size":
                dask_df = dask_df[dask_df.bulk_natoms <= val]
            else:
                warnings.warn("Filter is not implemented: " + name)
    return dask_df


def slab_filter(config, dask_df):
    slab_filters = config["slab_filters"]

    for name, val in slab_filters.items():
        if val != "None":
            if name == "filter_by_object_size":
                dask_df = dask_df[dask_df.slab_natoms <= val]
            else:
                warnings.warn("Filter is not implemented: " + name)
    return dask_df


def adsorbate_filter(config, dask_df):
    adsorbate_filters = config["adsorbate_filters"]

    for name, val in adsorbate_filters.items():
        if val != "None":
            if name == "filter_by_smiles":
                dask_df = dask_df[dask_df.adsorbate_smiles.isin(val)]
            else:
                warnings.warn("Filter is not implemented: " + name)

    return dask_df
