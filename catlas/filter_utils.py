import warnings
import numpy as np
from pymatgen.core.periodic_table import Element
import lmdb
import pickle
import os


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


def get_pourbaix_stability(mpid: str, conditions: dict) -> list:
    """Constructs a pourbaix diagram for the system of interest and evaluates the stability at desired points, returns a list of booleans capturing whether or not the material is stable under given conditions"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + "/pourbaix_diagrams/pbx1.lmdb"
    # Grab the entry of interest
    env = lmdb.open(
            str(path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=100,
        )
    str_en = mpid.encode("ascii")
    txn = env.begin()
    getit = txn.get(str_en)
    if getit == None:
        return [False]
    else:
        entry = pickle.loads(getit)
        env.close()
        
        # see what electrochemical conditions to consider and find the decomposition energies
        if set(("pH_lower", "pH_upper", "V_lower", "V_upper")).issubset(
            set(conditions.keys())
        ):
            decomp_bools = get_decomposition_bools_from_range(entry["pbx"], entry["pbx_entry"], conditions)
        elif "conditions" in conditions:
            decomp_bools = get_decomposition_bools_from_list(entry["pbx"], entry["pbx_entry"], conditions)
            return decomp_bools


def get_decomposition_bools_from_range(pbx, pbx_entry, conditions):
    """Evaluates the decomposition energies under the desired range of conditions"""
    list_of_bools = []

    # Use default step if one is not specified
    if "pH_step" not in conditions:
        conditions["pH_step"] = 0.2
    if "V_step" not in conditions:
        conditions["V_step"] = 0.1

    # Setup evaluation ranges
    pH_range = list(
        np.arange(conditions["pH_lower"], conditions["pH_upper"], conditions["pH_step"])
    )
    if conditions["pH_upper"] not in pH_range:
        pH_range.append(conditions["pH_upper"])

    V_range = list(
        np.arange(conditions["V_lower"], conditions["V_upper"], conditions["V_step"])
    )
    if conditions["V_upper"] not in V_range:
        V_range.append(conditions["V_upper"])

    # Iterate over ranges and evaluate bool outputs
    for pH in pH_range:
        for V in V_range:
            decomp_energy = pbx.get_decomposition_energy(pbx_entry, pH, V)
            if decomp_energy <= conditions["max_decomposition_energy"]:
                list_of_bools.append(True)
            else:
                list_of_bools.append(False)
    return list_of_bools


def get_decomposition_bools_from_list(pbx, pbx_entry, conditions):
    """Evaluates the decomposition energies under the desired set of conditions"""
    list_of_bools = []
    for condition in conditions["conditions"]:
        decomp_energy = pbx.get_decomposition_energy(
            pbx_entry, condition["pH"], condition["V"]
        )
        if decomp_energy <= conditions["max_decomposition_energy"]:
            list_of_bools.append(True)
        else:
            list_of_bools.append(False)
    return list_of_bools
