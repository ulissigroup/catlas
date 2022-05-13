import warnings
import numpy as np
from pymatgen.core.periodic_table import Element
import lmdb
import pickle
import os
import catlas
from mp_api import MPRester
import pickle
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.pourbaix_diagram import (
    PourbaixDiagram,
    PourbaixEntry,
)
from pymatgen.core import SETTINGS
import cerberus


def validate_config(yaml: dict) -> None:
    """
    Validates a config yaml for use in screening.
    """


def get_pourbaix_info(entry: dict) -> dict:
    """
    Grabs the relevant pourbaix entries for a given mpid
    from Materials Project and constructs a pourbaix diagram for it.

    Args:
        entry: bulk structure entry as constructed by
               catlas.load_bulk_structures.load_bulks_from_db
        mp_api_key: Users Materials Project API key (next-gen)

    """
    # Unpack mpid and define some presets
    pbx_entry = None
    mpid = entry["bulk_mpid"]
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
    with MPRester(SETTINGS["MP_API_KEY"]) as mpr:
        mpid_new = mpr.get_materials_id_from_task_id(mpid)
        pbx_entries = mpr.get_pourbaix_entries(comp)

        # Grab the pourbaix entry for the mpid of interest
        for entry in pbx_entries:
            if entry.entry_id == mpid:
                pbx_entry = entry

        # Handle the exception where our mpid is not the most up to date
        if pbx_entry == None:
            mpid_new = mpr.get_materials_id_from_task_id(mpid)
            for entry in pbx_entries:
                if entry.entry_id == mpid_new:
                    pbx_entry = entry

    # Construct pourbaix diagram
    pbx = PourbaixDiagram(pbx_entries, comp_dict=comp_dict, filter_solids=True)

    output["pbx"] = pbx
    output["pbx_entry"] = pbx_entry
    return output


def write_pourbaix_info(pbx_dicts: list, lmdb_path):
    """
    Writes the pourbaix query info to lmdb for future use.

    Args:
        pbx_dicts: list of dictionaries containing the important pourbaix query
                   info. Each dictionary contains pbx info for a single mpid
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

    for entry in pbx_dicts:
        # Add data and key values
        txn = db.begin(write=True)
        txn.put(
            key=entry["mpid"].encode("ascii"),
            value=pickle.dumps(entry, protocol=-1),
        )
        txn.commit()
    db.sync()
    db.close()


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


def get_pourbaix_stability(mpid: str, conditions: dict) -> list:
    """
    Reads the relevant Pourbaix info from the lmdb and evaluates stability at the desired
    points, returns a list of booleans capturing whether or not the material is stable
    under given conditions.

    Args:
        mpid: The mpid to perform stability analysis on
        conditions: The dictionary of Pourbaix settings set in the config yaml

    """
    lmdb_path = conditions["lmdb_path"]
    lmdb_path = "%s/%s" % (
        os.path.join(os.path.dirname(catlas.__file__), os.pardir),
        lmdb_path,
    )

    # Grab the entry of interest
    ## Open lmdb
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=100,
    )

    ## Begin read transaction (txn)
    txn = env.begin()

    ## Get entry and unpickle it
    entry_pickled = txn.get(mpid.encode("ascii"))
    entry = pickle.loads(entry_pickled)

    ## Close lmdb
    env.close()

    # Handle exception where pourbaix query was unsuccessful
    if entry["pbx"] == "failed":
        warnings.warn("Pourbaix data for " + mpid + " was not found.")
        return [False]

    # Determine electrochemical stability
    else:
        # see what electrochemical conditions to consider and find the decomposition energies
        if set(("pH_lower", "pH_upper", "V_lower", "V_upper")).issubset(
            set(conditions.keys())
        ):
            decomp_bools = get_decomposition_bools_from_range(
                entry["pbx"], entry["pbx_entry"], conditions
            )
        elif "conditions" in conditions:
            decomp_bools = get_decomposition_bools_from_list(
                entry["pbx"], entry["pbx_entry"], conditions
            )
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


from pymatgen.core import Element
from cerberus import Validator
import yaml

mpid_regex = "^mp-\d+$|^mvc-\d+$"  # 'mp-#' or 'mvc-#'
valid_element_groups = [
    "transition_metal",
    "post-transition metal",
    "metalloid",
    "rare earth metal",
    "alkali",
    "alkaline",
    "chalcogen",
    "halogen",
]


def validate_element(field, value, error):
    if not Element.is_valid_symbol(value):
        error(field, "'%s' is not an element" % value)


def validate_file_exists(field, value, error):
    if not os.path.exists(value):
        error(field, "file path does not exist: '%s'" % value)


schema = {
    "memory_cache_location": {"type": "string", "check_with": validate_file_exists},
    "input_options": {
        "required": True,
        "type": "dict",
        "schema": {
            "adsorbate_file": {
                "required": True,
                "type": "string",
                "check_with": validate_file_exists,
            },
            "bulk_file": {
                "required": True,
                "type": "string",
                "check_with": validate_file_exists,
            },
        },
    },
    "adsorbate_filters": {
        "type": "dict",
        "schema": {"filter_by_smiles": {"type": "list", "schema": {"type": "string"}}},
    },
    "bulk_filters": {
        "type": "dict",
        "schema": {
            "filter_by_mpids": {"type": "list", "regex": mpid_regex},
            "filter_ignore_mpids": {"type": "list", "regex": mpid_regex},
            "filter_by_acceptable_elements": {
                "type": "list",
                "check_with": validate_element,
            },
            "filter_by_required_elements": {
                "type": "list",
                "check_with": validate_element,
            },
            "filter_by_num_elements": {"type": "integer"},
            "filter_by_object_size": {"type": "integer"},
            "filter_by_elements_active_host": {
                "type": "dict",
                "schema": {
                    "active": {"type": "string", "check_with": validate_element},
                    "host": {"type": "string", "check_with": validate_element},
                },
            },
            "filter_by_element_groups": {
                "type": "list",
                "allowed": valid_element_groups,
            },
            "filter_by_pourbaix_stability": {
                "type": "dict",
                "schema": {
                    "lmdb_path": {
                        "required": True,
                        "type": "string",
                        "check_with": validate_file_exists,
                    },
                    "conditions": {
                        "required": True,
                        "excludes": "pH_lower",
                        "type": "dict",
                        "schema": {
                            "pH": {"type": "float"},
                            "V": {"type": "float"},
                        },
                    },
                    "pH_lower": {
                        "required": True,
                        "excludes": "conditions",
                        "dependencies": ["pH_upper", "V_lower", "V_upper"],
                    },
                    "pH_step": {
                        "type": "float",
                        "dependencies": "pH_lower",
                    },
                    "V_step": {"type": "float", "dependencies": "pH_lower"},
                },
            },
        },
    },
    "slab_filters": {
        "type": "dict",
        "schema": {
            "filter_by_object_size": {"type": "integer"},
            "filter_by_max_miller_index": {"type": "integer"},
        },
    },
    "output_options": {
        "required": True,
        "type": "dict",
        "schema": {
            "make_parity_plots": {"type": "boolean"},
            "output_all_structures": {"type": "boolean"},
            "pickle_intermediate_outputs": {"type": "boolean"},
            "pickle_final_output": {"type": "boolean"},
            "verbose": {"type": "boolean"},
            "run_name": {"required": True, "type": "string"},
        },
    },
    "adslab_prediction_steps": {
        "required": False,
        "type": "list",
        "schema": {
            "type": "dict",
            "schema": {
                "checkpoint_path": {
                    "type": "string",
                    "check_with": validate_file_exists,
                    "regex": ".*.pt",  # cerberus doesn't understand re "$"; requires full match
                },
                "gpu": {"type": "boolean", "required": True},
                "label": {
                    "required": True,
                    "type": "string",
                },
                "number_steps": {
                    "type": "integer",
                },
                "batch_size": {"type": "integer"},
            },
        },
    },
}

config_validator = Validator(schema)
