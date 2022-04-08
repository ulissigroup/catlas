import warnings
import numpy as np
from pymatgen.core.periodic_table import Element
from catlas.filter_utils import get_pourbaix_stability, get_elements_in_groups
from catlas.sankey.sankey_utils import update_dictionary


def bulk_filter(config, dask_df, sankey_dict, initial_bulks):
    """
    Filters a dask dataframe `dask_df` of bulk structures according to rules specified in a config yml `config`.

    Args:
        config: dictionary of the config yaml
        dask_df: the working dataframe of bulk inputs
        sankey_dict: a dictionary of values that will be used to populate the output sankey diagram
        initial_bulks: the initial number of bulks

    Returns:
        dask_df: the working dataframe of bulk values post-filtering
        sankey_dict: the sankey dictionary with bulk filtering information included
    """
    bulk_filters = config["bulk_filters"]
    sankey_idx = 2
    sankey_dict["label"][0] = f"Bulks from db ({initial_bulks})"

    for name, val in bulk_filters.items():
        if (
            str(val) != "None"
        ):  # depending on how yaml is specified, val may either be "None" or NoneType
            if name == "filter_by_mpids":
                dask_df = dask_df[dask_df.bulk_mpid.isin(val)]
            elif name == "filter_ignore_mpids":
                dask_df = dask_df[~dask_df.bulk_mpid.isin(val)]
            elif name == "filter_by_acceptable_elements":
                dask_df = dask_df[
                    dask_df.bulk_elements.apply(
                        lambda x, valid_elements: all(
                            [el in valid_elements for el in x]
                        ),
                        valid_elements=val,
                        meta=("bulk_elements", "bool"),
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
            elif name == "filter_by_elements_active_host":
                dask_df = dask_df[
                    dask_df.bulk_elements.apply(
                        lambda x, active, host: all(
                            [
                                all([el in [*active, *host] for el in x]),
                                any([el in host for el in x]),
                                any([el in active for el in x]),
                            ]
                        ),
                        active=val["active"],
                        host=val["host"],
                        meta=("bulk_elements", "bool"),
                    )
                ]
            elif name == "filter_by_element_groups":
                valid_els = get_elements_in_groups(val)
                dask_df = dask_df[
                    dask_df.bulk_elements.apply(
                        lambda x, valid_elements: all(
                            [el in valid_elements for el in x]
                        ),
                        valid_elements=valid_els,
                        meta=("bulk_elements", "bool"),
                    )
                ]
            elif name == "filter_by_pourbaix_stability":
                dask_df = dask_df[
                    dask_df.bulk_mpid.apply(
                        lambda x, conditions: any(
                            get_pourbaix_stability(x, conditions)
                        ),
                        conditions=val,
                        meta=("bulk_mpid", "bool"),
                    )
                ]

            else:
                warnings.warn("Bulk filter is not implemented: " + name)

            if config["output_options"]["verbose"]:
                print(
                    'filter "'
                    + name
                    + '" filtered bulk df down to length '
                    + str(len(dask_df))
                )
            # Update the sankey dictionary
            node_loss = initial_bulks - len(dask_df)
            initial_bulks = len(dask_df)
            sankey_dict = update_dictionary(
                sankey_dict,
                f"Rejected by {name} ({node_loss})",
                0,
                sankey_idx,
                node_loss,
            )
            sankey_idx += 1
        sankey_dict = update_dictionary(
            sankey_dict,
            f"Filtered bulks ({initial_bulks})",
            0,
            sankey_idx,
            initial_bulks,
        )
    return dask_df, sankey_dict


def slab_filter(config, dask_dict):
    """Filters a dask bag `dask_dict` according to rules specified in config yml `config`"""
    slab_filters = config["slab_filters"]

    keep = True

    for name, val in slab_filters.items():
        if val != "None":
            if name == "filter_by_object_size":
                keep = keep and (dask_dict["slab_natoms"] <= val)
            elif name == "filter_by_max_miller_index":
                keep = keep and (dask_dict["slab_max_miller_index"] <= val)
            else:
                warnings.warn("Slab filter is not implemented: " + name)
    return keep


def adsorbate_filter(config, dask_df):
    """Filters a dask dataframe `dask_df` of adsorbate structures according to rules specified in config yml `config`"""
    adsorbate_filters = config["adsorbate_filters"]

    for name, val in adsorbate_filters.items():
        if val != "None":
            if name == "filter_by_smiles":
                dask_df = dask_df[dask_df.adsorbate_smiles.isin(val)]
            else:
                warnings.warn("Adsorbate filter is not implemented: " + name)

    return dask_df
