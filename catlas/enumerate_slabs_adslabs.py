import copy
import logging
import numpy as np

import torch
from catlas.enumeration_utils import enumerate_surfaces_for_saving
from ocdata.adsorbates import Adsorbate
from ocdata.bulk_obj import Bulk
from ocdata.combined import Combined
from ocdata.surfaces import Surface
from ocpmodels.preprocessing import AtomsToGraphs
from pymatgen.io.ase import AseAtomsAdaptor


class CustomAdsorbate(Adsorbate):
    """A custom adsorbate object for ocdata compatability."""

    def __init__(self, ads_atoms, bond_indices, smiles):
        self.atoms = ads_atoms
        self.bond_indices = bond_indices
        self.smiles = smiles


class CustomBulk(Bulk):
    """A custom bulk object for ocdata compatability."""

    def __init__(self, bulk_atoms):
        self.bulk_atoms = bulk_atoms


def enumerate_slabs(bulk_dict, max_miller):
    """
    Given a dictionary defining a material bulk, use pymatgen's SlabGenerator object
    to enumerate all the slabs associated with the bulk object.
    Args:
        bulk_dict (dict): A dictionary containing a key "bulk_structure" containing
            an pymatgen structure object corresponding to a bulk material.
        max_miller (int, optional): The highest miller index to enumerate up to.
    Returns:
        list[dict]: A list of dictionaries corresponding to the surfaces enumerated
            from the input bulk. Each dictionary has the following key-value pairs:
            -- slab_surface_object (ocdata.surfaces.Surface): the ocdata surface object.
            -- slab_millers (tuple[int]): the miller index of the surface.
            -- slab_max_miller_index (int): the highest miller index of the surface.
            -- slab_shift (float): the shift of the termination of the surface.
            -- slab_top (bool): whether the termination is the top of the slab.
    """
    bulk_dict = copy.deepcopy(bulk_dict)

    logger = logging.getLogger("distributed.worker")
    logger.info("enumerate_slabs_started: %s" % str(bulk_dict))

    bulk_obj = CustomBulk(AseAtomsAdaptor.get_atoms(bulk_dict["bulk_structure"]))

    surfaces = enumerate_surfaces_for_saving(bulk_dict["bulk_structure"], max_miller)
    surface_list = []
    for surface in surfaces:
        surface_struct, millers, shift, top = surface
        surface_object = Surface(bulk_obj, surface, np.nan, np.nan)

        # Make the surface dict by extending a copy of the bulk dict
        surface_result = copy.deepcopy(bulk_dict)
        surface_result.update(
            {
                "slab_surface_object": surface_object,
                "slab_millers": millers,
                "slab_max_miller_index": np.max(millers),
                "slab_shift": shift,
                "slab_top": top,
                "slab_natoms": len(surface_object.surface_atoms),
                "slab_structure": surface_struct,
            }
        )
        surface_list.append(surface_result)

    logger.info("enumerate_slabs_finished: %s" % str(bulk_dict))

    return surface_list


def enumerate_adslabs(surface_ads_combo):
    """
    Generate adslabs from two dictionaries specifying a surface and an adsorbate. The
    only difference between these adslabs is the surface site where the adsorbate binds,
    allowing every adslab to be a shallow copy of the same object with updated positions.
    Because of this, be careful about updating adslab properties!

    Args:
        surface_ads_combo ([dict, dict]): a list containing a surface dictionary and an
            adsorbate dictionary.
    Returns:
        list[ase.Atoms]: A list of Atoms objects of the adslab systems with constraints
            applied.
    """
    surface_dict, ads_dict = copy.deepcopy(surface_ads_combo)

    # Prep the new adslab result from the adsorbate and surface info dicts
    adslab_result = []

    adsorbate_obj = CustomAdsorbate(
        ads_dict["adsorbate_atoms"],
        ads_dict["adsorbate_bond_indices"],
        ads_dict["adsorbate_smiles"],
    )
    combo_obj = Combined(
        adsorbate_obj, surface_dict["slab_surface_object"], enumerate_all_configs=True
    )

    adslab_result = combo_obj.constrained_adsorbed_surfaces

    """The adslab_result object takes up a ton of memory because they are all copies of
    the same atoms object. To reduce memory use, let's just store a shallow copy of the
    atoms for each, and update the positions. Note that this means all other atoms
    properties are shared between copies, so be careful if you need to modify things!"""
    adslab_result_shallow_copy = []
    for atoms in adslab_result:
        atoms_copy = copy.copy(adslab_result[0])
        atoms_copy.arrays = copy.copy(atoms_copy.arrays)
        atoms_copy.arrays["positions"] = atoms.positions
        adslab_result_shallow_copy.append(atoms_copy)

    return adslab_result_shallow_copy


def convert_adslabs_to_graphs(adslab_result, max_neighbors=50, cutoff=6):
    """
    Turn ase.Atoms adslabs into graphs compatible with ocp models.

    Args:
        adslab_result (ase.Atoms): an ase.Atoms object containing an adsorbate
            positioned on top of a surface.
        max_neighbors (int, optional): The highest number of neighbors to be
            considered in a graph. If a node ends up with more than this many neighbors,
            the furthest neighbors will be ignored. Defaults to 50.
        cutoff (int, optional): The maximum distance in Angstroms to look for neighbors.
            Defaults to 6.

    Returns:
        graph_dict (dict): A dictionary containing a single key "adslab_graphs" which
            contains a list of torch_geometric.data.Data objects that can be used by OCP models.
    """
    adslab_result = copy.deepcopy(adslab_result)

    graph_dict = {}

    a2g = AtomsToGraphs(
        max_neigh=max_neighbors,
        radius=cutoff,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=False,
    )

    graph_list = []

    for adslab in adslab_result:
        tags = adslab.get_tags()
        graph = a2g.convert(adslab)
        graph.tags = torch.LongTensor(tags)
        graph.fid = 0
        graph.sid = 0
        graph_list.append(graph)

    graph_dict["adslab_graphs"] = graph_list

    return graph_dict


def merge_surface_adsorbate_combo(surface_adsorbate_combo):
    """
    Combines a surface dict and an adsorbate dict.

    Args:
        surface_adsorbate_combo (Iterable[dict, dict]): a surface and an adsorbate.

    Returns:
        dict: a single dictionary containing a surface and an adsorbate.
    """
    surface_dict, ads_dict = copy.deepcopy(surface_adsorbate_combo)

    # Prep the new adslab result from the adsorbate and surface info dicts
    adslab_result = {}
    adslab_result.update(surface_dict)
    adslab_result.update(ads_dict)

    return adslab_result
