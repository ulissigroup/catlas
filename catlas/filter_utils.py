import os
import pickle
import warnings

import lmdb
import numpy as np
from mp_api import MPRester

import catlas
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor


def get_pourbaix_info(entry: dict) -> dict:
    """
    Grabs the relevant pourbaix entries for a given mpid
    from Materials Project and constructs a pourbaix diagram for it. This currently
    only supports MP materials.

    Args:
        entry: bulk structure entry as constructed by
               catlas.load_bulk_structures.load_bulks_from_db

    """
    # Unpack mpid and define some presets
    pbx_entry = None
    mpid = entry["bulk_id"]

    # Raise an error if non-MP materials used
    if mpid.split("-")[0] != "mp" and mpid.split("-")[0] != "mvc":
        raise ValueError(
            """Pourbaix filtering is only supported for Materials Project materials (bad
            id: '%s')."""
            % mpid
        )

    output = {"mpid": mpid}

    # Get the composition information
    pmg_entry = AseAtomsAdaptor.get_structure(entry["bulk_atoms"])
    comp_dict = pmg_entry.composition.fractional_composition.as_dict()
    comp_dict.pop("H", None)
    comp_dict.pop("O", None)
    comp = list(comp_dict.keys())

    # Handle exception where material is pure H / O
    if len(comp) == 0:
        output["pbx"] = "failed"
        output["pbx_entry"] = "failed"
        return output

    # Grab the Pourbaix entries
    with MPRester() as mpr:
        mpid_new = mpr.get_materials_id_from_task_id(mpid)

        # Handle the exception where the material has been fully deprecated
        if mpid_new is None:
            output["pbx"] = "failed"
            output["pbx_entry"] = "failed"
            return output

        pbx_entries = mpr.get_pourbaix_entries(comp)

        # Grab the pourbaix entry for the mpid of interest
        for entry in pbx_entries:
            if entry.entry_id == mpid:
                pbx_entry = entry

        # Handle the exception where our mpid is not the most up to date
        if pbx_entry is None:
            mpid_new = mpr.get_materials_id_from_task_id(mpid)
            for entry in pbx_entries:
                if entry.entry_id == mpid_new:
                    pbx_entry = entry

    # Construct pourbaix diagram
    pbx = PourbaixDiagram(pbx_entries, comp_dict=comp_dict, filter_solids=True)

    output["pbx"] = pbx
    output["pbx_entry"] = pbx_entry
    return output


def write_pourbaix_info(pbx_entry: dict, lmdb_path):
    """
    Writes the pourbaix query info to lmdb for future use.

    Args:
        pbx_entry: dictionary containing the important pourbaix query
                   info info for a single mpid
        lmdb_path: Location where the lmdb will be written (including fname)

    """

    lmdb_path = "%s/%s" % (
        os.path.join(os.path.dirname(catlas.__file__), os.pardir),
        lmdb_path,
    )

    db = lmdb.open(
        lmdb_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # Add data and key values
    txn = db.begin(write=True)
    txn.put(
        key=pbx_entry["mpid"].encode("ascii"),
        value=pickle.dumps(pbx_entry, protocol=-1),
    )
    txn.commit()
    db.sync()
    db.close()


def pb_query_and_write(entry: dict, lmdb_path: str):
    """Pull pourbaix info from MP and writes it to the lmdb"""
    pourbaix_info = get_pourbaix_info(entry)
    write_pourbaix_info(pourbaix_info, lmdb_path)


def get_elements_in_groups(groups: list) -> list:
    """Grabs the element symbols of all elements in the specified groups"""
    valid_els = []

    if "transition metal" in groups:
        new_valid_els = [str(el) for el in Element if el.is_transition_metal]
        valid_els.extend(new_valid_els)
    if "post-transition metal" in groups:
        new_valid_els = [str(el) for el in Element if el.is_post_transition_metal]
        valid_els.extend(new_valid_els)
    if "metalloid" in groups:
        new_valid_els = [str(el) for el in Element if el.is_metalloid]
        valid_els.extend(new_valid_els)
    if "rare earth metal" in groups:
        new_valid_els = [str(el) for el in Element if el.is_rare_earth_metal]
        valid_els.extend(new_valid_els)
    if "alkali" in groups:
        new_valid_els = [str(el) for el in Element if el.is_alkali]
        valid_els.extend(new_valid_els)
    if "alkaline" in groups or "alkali earth" in groups:
        new_valid_els = [str(el) for el in Element if el.is_alkaline]
        valid_els.extend(new_valid_els)
    if "chalcogen" in groups:
        new_valid_els = [str(el) for el in Element if el.is_calcogen]
        valid_els.extend(new_valid_els)
    if "halogen" in groups:
        new_valid_els = [str(el) for el in Element if el.is_halogen]
        valid_els.extend(new_valid_els)

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


def get_pourbaix_stability(entry: dict, conditions: dict) -> list:
    """
    Reads the relevant Pourbaix info from the lmdb and evaluates stability at the
    desired points, returns a list of booleans capturing whether or not the material is
    stable under given conditions.

    Args:
        entry: A dictionary containing the bulk entry which will be assessed
        conditions: The dictionary of Pourbaix settings set in the config yaml

    """
    lmdb_path_simple = conditions["lmdb_path"]
    bulk_id = entry["bulk_id"]
    lmdb_path = "%s/%s" % (
        os.path.join(os.path.dirname(catlas.__file__), os.pardir),
        lmdb_path_simple,
    )

    # Grab the entry of interest
    # Open lmdb
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=100,
    )

    # Begin read transaction (txn)
    txn = env.begin()

    # Get entry and unpickle it
    entry_pickled = txn.get(bulk_id.encode("ascii"))

    if entry_pickled is not None:
        entry = pickle.loads(entry_pickled)
    else:
        print(f"querying {bulk_id} becuase it wasn't found in the lmdb path provided")
        env.close()
        pb_query_and_write(entry, lmdb_path_simple)
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=100,
        )
        txn = env.begin()
        entry_pickled = txn.get(bulk_id.encode("ascii"))
        entry = pickle.loads(entry_pickled)

    # Close lmdb
    env.close()

    # Handle exception where pourbaix query was unsuccessful
    if entry["pbx"] == "failed":
        warnings.warn("Pourbaix data for " + bulk_id + " was not found.")
        return [False]
    # Determine electrochemical stability
    else:
        # see what electrochemical conditions to consider and find the decomposition
        # energies
        if "pH_lower" in conditions.keys():
            decomp_bools = get_decomposition_bools_from_range(
                entry["pbx"], entry["pbx_entry"], conditions
            )
        elif "conditions_list" in conditions.keys():
            decomp_bools = get_decomposition_bools_from_list(
                entry["pbx"], entry["pbx_entry"], conditions, bulk_id
            )
        return decomp_bools


def get_decomposition_bools_from_range(pbx, pbx_entry, conditions):
    """Evaluates decomposition energies at regular pH and voltage windows within
    specified intervals.

    Args:
        pbx (pymatgen.analysis.pourbaix_diagram.PourbaixDiagram): a pourbaix diagram
        object containing information about the reference chemical system.
        pbx_entry (pymatgen.analysis.pourbaix_diagram.PourbaixEntry): a pourbaix entry
        specific to the material we want to calculate the decomposition entry of.
        conditions (dict): A dictionary specifying what conditions to evaluate the
        decomposition energy at. Contains the following key-value pairs:
            pH_step (float): how far apart to evaluate consecutive pH points
            pH_upper (float): the maximum pH to evaluate
            pH_lower (float) the minimum pH to evaluate
            V_step (float): how far apart to evaluate consecutive voltage points
            V_upper (float): the maximum voltage to evaluate
            V_lower (float) the minimum voltage to evaluate

    Returns:
        Iterable[bool]: A list corresponding to whether the input entry is stable under
        each set of conditions.
    """
    list_of_bools = []

    # Use default step if one is not specified
    if "pH_step" not in conditions:
        conditions["pH_step"] = 0.2
    if "V_step" not in conditions:
        conditions["V_step"] = 0.1
    max_decomposition_energy = conditions["max_decomposition_energy"]

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
            if decomp_energy <= max_decomposition_energy:
                list_of_bools.append(True)
            else:
                list_of_bools.append(False)
    return list_of_bools


def get_decomposition_bools_from_list(pbx, pbx_entry, conditions, bulk_id):
    """Evaluates decomposition energies at regular pH and voltage windows at specified pH and voltage points.

    Args:
        pbx (pymatgen.analysis.pourbaix_diagram.PourbaixDiagram): a pourbaix diagram
        object containing information about the reference chemical system.
        pbx_entry (pymatgen.analysis.pourbaix_diagram.PourbaixEntry): a pourbaix entry
        specific to the material we want to calculate the decomposition entry of.
        conditions (list[dict]): A list of dictionaries specifying what conditions to
        evaluate the decomposition energy at. Each dictionary contains the following
        key-value pairs:
            pH (float): the pH to evaluate a decomposition at.
            V (float): the voltage to evaluate a voltage at.

    Returns:
        Iterable[bool]: A list corresponding to whether the input entry is stable under
        each set of conditions.
    """
    list_of_bools = []
    for condition in conditions["conditions_list"]:
        decomp_energy = pbx.get_decomposition_energy(
            pbx_entry, condition["pH"], condition["V"]
        )
        if decomp_energy <= condition["max_decomposition_energy"]:
            list_of_bools.append(True)
        else:
            list_of_bools.append(False)
    return list_of_bools


def get_first_type(x):
    """Get the type of the input, unpacking lists first if necessary."""
    if type(x) == list and len(x) > 0:
        return type(x[0])
    else:
        return type(x)
