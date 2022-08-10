import os
import pickle
import warnings

import cerberus
import lmdb
import numpy as np
from mp_api import MPRester

import catlas
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixEntry
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def get_pourbaix_info(entry: dict) -> dict:
    """
    Grabs the relevant pourbaix entries for a given mpid
    from Materials Project and constructs a pourbaix diagram for it. This currently only supports MP materials.

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
            "Pourbaix filtering is only supported for Materials Project materials (bad id: '%s')."
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
    Reads the relevant Pourbaix info from the lmdb and evaluates stability at the desired
    points, returns a list of booleans capturing whether or not the material is stable
    under given conditions.

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
    entry_pickled = txn.get(bulk_id.encode("ascii"))

    if entry_pickled != None:
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

    ## Close lmdb
    env.close()

    # Handle exception where pourbaix query was unsuccessful
    if entry["pbx"] == "failed":
        warnings.warn("Pourbaix data for " + bulk_id + " was not found.")
        return [False]
    # Determine electrochemical stability
    else:
        # see what electrochemical conditions to consider and find the decomposition energies
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
    """Evaluates the decomposition energies under the desired range of conditions"""
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
    """Evaluates the decomposition energies under the desired set of conditions"""
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
    if type(x) == list and len(x) > 0:
        return type(x[0])
    else:
        return type(x)

    
def add_full_wyckoffs(bulk):

    sg = SpacegroupAnalyzer(bulk)
    symbulk = sg.get_symmetrized_structure()
    wyckoff_letters = symbulk.wyckoff_letters
    equivalent_indices = symbulk.equivalent_indices

    wyckoffs = []
    for i, site in enumerate(symbulk):
        eqi = [eqi for eqi in equivalent_indices if i in eqi][0]
        wyckoffs.append('%s%s%s' %(site.species_string, len(eqi), symbulk.wyckoff_letters[i]))
    symbulk.add_site_property('full_wyckoff', wyckoffs)
    return symbulk

def surface_area(slab):
    m = slab.lattice.matrix
    return np.linalg.norm(np.cross(m[0], m[1]))


def get_bond_length(ucell, neighbor_factor):
    
    bond_lengths_dict = {}
    for site in ucell:
        if site.full_wyckoff in bond_lengths_dict.keys():
            continue
        r, neighbors = 2, []
        while len(neighbors) == 0:
            neighbors = ucell.get_neighbors(site, r)
            r += 1
        neighbors = sorted(neighbors, key=lambda n: n[1])
        d = neighbors[0]
        bond_lengths_dict[site.full_wyckoff] = d[1]*neighbor_factor
        
    return bond_lengths_dict


def get_bulk_cn(ucell, neighbor_factor):
    bond_lengths_dict = get_bond_length(ucell, neighbor_factor)
    bulk_cn_dict = {}
    for site in ucell:
        k = site.full_wyckoff
        if k in bulk_cn_dict.keys():
            continue
        bulk_cn_dict[k] = len(ucell.get_neighbors(site, bond_lengths_dict[k]))
    return bulk_cn_dict


def get_total_bb(dask_dict: dict, neighbor_factor: float) -> float:
    ucell = AseAtomsAdaptor.get_structure(dask_dict["bulk_atoms"])
    bulk_cn_dict = get_bulk_cn(ucell, neighbor_factor)
    bind_length_dict = get_bond_length(ucell, neighbor_factor)
    tot_bb = 0
    for site in AseAtomsAdaptor.get_structure(dask_dict["slab_surface_object"].surface_atoms):
        if site.frac_coords[2] < slab.center_of_mass[2]:
            # analyze the top surface only
            continue
        neighbors = slab.get_neighbors(site, bind_length_dict[site.full_wyckoff])
        bulk_cn = bulk_cn_dict[site.full_wyckoff]
        if len(neighbors) == bulk_cn:
            continue
        if len(neighbors) > bulk_cn:
            warnings.warn(f"For {dask_dict["bulk_id"]} {dask_dict["slab_millers"]} {dask_dict["slab_shift"]} the slab cn was observed to be larger than the bulk")
        tot_bb += (bulk_cn - len(neighbors)) / bulk_cn
    return tot_bb


def get_total_nn(dask_dict: dict, neighbor_factor: float) -> int:
    ucell = AseAtomsAdaptor.get_structure(dask_dict["bulk_atoms"])
    bulk_cn_dict = get_bulk_cn(ucell, neighbor_factor)
    bind_length_dict = get_bond_length(ucell, neighbor_factor)
    tot_nn = 0
    for site in AseAtomsAdaptor.get_structure(dask_dict["slab_surface_object"].surface_atoms):
        if site.frac_coords[2] < slab.center_of_mass[2]:
            # analyze the top surface only
            continue
        neighbors = slab.get_neighbors(site, bind_length_dict[site.full_wyckoff])
        bulk_cn = bulk_cn_dict[site.full_wyckoff]
        if len(neighbors) == bulk_cn:
            continue
        if len(neighbors) > bulk_cn:
            warnings.warn(f"For {dask_dict["bulk_id"]} {dask_dict["slab_millers"]} {dask_dict["slab_shift"]} the slab cn was observed to be larger than the bulk")
        tot_nn += len(neighbors)
    return tot_nn


def get_broken_bonds(dask_dict: dict, neighbor_factor: float) -> float:
    a = surface_area(slab)
    cns = get_total_bb(ucell, slab, neighbor_factor)
    return cns * ecoh * (1 / (2 * a))

def get_surface_density(dask_dict: dict, neighbor_factor: float) -> float:
    a = surface_area(slab)
    cns = get_total_nn(dask_dict, neighbor_factor)
    return cns * ecoh * (1 / (2 * a))