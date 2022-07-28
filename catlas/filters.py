import warnings
import numpy as np
from pymatgen.core.periodic_table import Element
from catlas.filter_utils import get_pourbaix_stability, get_elements_in_groups
import catlas.cache_utils
import catlas.dask_utils


def bulk_filter(config, dask_df, sankey=None, initial_bulks=None):
    """
    Filters a dask dataframe `dask_df` of bulk structures according to rules specified in a config yml `config`.

    Args:
        config: dictionary of the config yaml
        dask_df: the working dataframe of bulk inputs
        sankey: the sankey object
        initial_bulks: the initial number of bulks

    Returns:
        dask_df: the working dataframe of bulk values post-filtering
        sankey: the sankey object with added info
    """
    bulk_filters = config["bulk_filters"]
    columns = dask_df.columns

    if sankey is not None:
        sankey_idx = 2
        sankey.info_dict["label"][1] = f"Bulks from db ({initial_bulks})"

    for name, val in bulk_filters.items():
        if (
            str(val) != "None"
        ):  # depending on how yaml is specified, val may either be "None" or NoneType
            if name == "filter_by_bulk_ids" and "bulk_id" in columns:
                dask_df = dask_df[dask_df.bulk_id.isin(val)]
            elif name == "filter_ignore_bulk_ids":
                dask_df = dask_df[~dask_df.bulk_id.isin(val)]
            elif name == "filter_by_acceptable_elements":
                dask_df = dask_df[
                    dask_df.bulk_elements.apply(
                        lambda x, valid_elements: all(
                            [el in valid_elements for el in x]
                        ),
                        valid_elements=val,
                        meta=("in_acceptable_elements", "bool"),
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
                        meta=("active_host_satisfied", "bool"),
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
                        meta=("element_in_group", "bool"),
                    )
                ]
            elif name == "filter_by_pourbaix_stability":
                dask_df = dask_df[
                    dask_df.apply(
                        lambda x, conditions: any(
                            get_pourbaix_stability(x, conditions)
                        ),
                        axis=1,
                        conditions=val,
                        meta=("pourbaix_stability", "bool"),
                    )
                ]

            elif name == "filter_by_bulk_e_above_hull":
                dask_df = dask_df[dask_df.bulk_e_above_hull <= val]

            elif name == "filter_by_bulk_band_gap":
                if "min_gap" in val and "max_gap" in val:
                    dask_df = dask_df[
                        (dask_df.bulk_band_gap >= val["min_gap"])
                        & (dask_df.bulk_band_gap <= val["max_gap"])
                    ]
                elif "min_gap" in val:
                    dask_df = dask_df[(dask_df.bulk_band_gap >= val["min_gap"])]
                elif "max_gap" in val:
                    dask_df = dask_df[(dask_df.bulk_band_gap <= val["max_gap"])]
                else:
                    warnings.warn(
                        "Band gap filtering was not specified properly -> skipping it."
                    )

            else:
                warnings.warn("Bulk filter is not implemented: " + name)

            if config["output_options"]["verbose"]:
                print(
                    'filter "'
                    + name
                    + '" filtered bulk df down to length '
                    + str(len(dask_df))
                )
            if sankey is not None:
                # Update the sankey dictionary
                node_loss = initial_bulks - len(dask_df)
                initial_bulks = len(dask_df)
                sankey.update_dictionary(
                    f"Rejected by {name} ({node_loss})",
                    1,
                    sankey_idx,
                    node_loss,
                    1,
                    "tbd",
                )
                sankey_idx += 1
    if sankey is not None:
        # Add remaining bulks to the sankey and connect them to slabs
        sankey.update_dictionary(
            f"Filtered bulks ({initial_bulks})",
            1,
            sankey_idx,
            initial_bulks,
            0.2,
            0.5,
        )
        sankey.update_dictionary(
            "Slabs",
            sankey_idx,
            sankey_idx + 1,
            initial_bulks,
            0.4,
            0.5,
        )
        return dask_df, sankey
    else:
        return dask_df


def slab_filter(config, dask_dict):
    """
    Filters a dask bag `dask_dict` of slabs according to rules specified in config yml `config`
        Args:
        config: dictionary of the config yaml
        dask_dict: a dictionary containing slab info

    Returns:
        boolean value (True -> retain slab, False -> reject slab)
    """
    slab_filters = config["slab_filters"]

    keep = True

    for name, val in slab_filters.items():
        if val != "None":
            if name == "filter_by_object_size":
                keep = keep and (dask_dict["slab_natoms"] <= val)
            elif name == "filter_by_max_miller_index":
                keep = keep and (np.abs(dask_dict["slab_max_miller_index"]) <= val)
            else:
                warnings.warn("Slab filter is not implemented: " + name)
    return keep


def adsorbate_filter(config, dask_df, sankey):
    """
    Filters a dask dataframe `dask_df` of adsorbate structures according to rules specified in config yml `config`.
    Args:
        config: dictionary of the config yaml
        dask_df: the working dataframe of adsorbate inputs
        sankey: the sankey object
        initial_bulks: the initial number of bulks

    Returns:
        dask_df: the working dataframe of adsorbate values post-filtering
        sankey: the sankey object with added information
    """
    adsorbate_filters = config["adsorbate_filters"]
    initial_adsorbate = len(dask_df)
    sankey.info_dict["label"][0] = f"Adsorbates from db ({initial_adsorbate})"

    for name, val in adsorbate_filters.items():
        if val != "None":
            if name == "filter_by_smiles":
                dask_df = dask_df[dask_df.adsorbate_smiles.isin(val)]

    # Update the sankey diagram
    node_idx = len(sankey.info_dict["label"])
    sankey.update_dictionary(
        f"Filtered adsorbates ({len(dask_df)})",
        0,
        node_idx,
        len(dask_df),
        0.2,
        0.2,
    )

    sankey.update_dictionary(
        f"Rejected adsorbates ({initial_adsorbate - len(dask_df)})",
        0,
        node_idx + 1,
        initial_adsorbate - len(dask_df),
        1,
        0.001,
    )
    sankey.update_dictionary(
        "Adslabs", node_idx, len(sankey.info_dict["label"]), len(dask_df), 0.8, 0.25
    )
    return dask_df, sankey
