import warnings
import numpy as np
from pymatgen.core.periodic_table import Element


def bulk_filter(config, dask_df):
    """Filters a dask dataframe `dask_df` of bulk structures according to rules specified in a config yml `config`"""
    bulk_filters = config["bulk_filters"]

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

            else:
                warnings.warn("Bulk filter is not implemented: " + name)

            if config["output_options"]["verbose"]:
                print(
                    'filter "'
                    + name
                    + '" filtered bulk df down to length '
                    + str(len(dask_df))
                )
    return dask_df


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


def get_elements_in_groups(groups: list) -> list:
    """Grabs the element symbols of all elements in the specified groups"""
    valid_els = []

    if "transition metal" in groups:
        new_valid_els = [str(el) for el in Element if el.is_transition_metal]
        valid_els = [*valid_els, new_valid_els]
    if "post-transition metal" in groups:
        new_valid_els = [str(el) for el in Element if el.is_post_transition_metal]
        valid_els = [*valid_els, new_valid_els]
    if "metalloid" in groups:
        new_valid_els = [str(el) for el in Element if el.is_metalloid]
        valid_els = [*valid_els, new_valid_els]
    if "rare earth metal" in groups:
        new_valid_els = [str(el) for el in Element if el.is_rare_earth_metal]
        valid_els = [*valid_els, new_valid_els]
    if "alkali" in groups:
        new_valid_els = [str(el) for el in Element if el.is_alkali]
        valid_els = [*valid_els, new_valid_els]
    if "alkaline" in groups or "alkali earth" in groups:
        new_valid_els = [str(el) for el in Element if el.is_alkaline]
        valid_els = [*valid_els, new_valid_els]
    if "chalcogen" in groups:
        new_valid_els = [str(el) for el in Element if el.is_calcogen]
        valid_els = [*valid_els, new_valid_els]
    if "halogen" in groups:
        new_valid_els = [str(el) for el in Element if el.is_halogen]
        valid_els = [*valid_els, new_valid_els]

    implemented_groups = [
        "transition metal",
        "post-transition metal",
        "metalloid",
        "rare earth metal",
        "alkali",
        "alkaline",
        "alkali earth",
        "chalcogen",
        "halogen",
    ]

    for group in groups:
        if group not in implemented_groups:
            warnings.warn(
                "Group not implemented: "
                + group
                + "\n Implemented groups are: "
                + str(implemented_groups)
            )
    return list(np.unique(valid_els))
