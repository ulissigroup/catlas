import numpy as np
from ocdata import precompute_sample_structures as compute
from ocdata.surfaces import Surface
from ocdata.combined import Combined
from ocdata.adsorbates import Adsorbate
from ocdata.bulk_obj import Bulk
import copy
from ocpmodels.preprocessing import AtomsToGraphs
import logging
import torch
import catlas.dask_utils


class CustomAdsorbate(Adsorbate):
    def __init__(self, ads_atoms, bond_indicies, smiles):
        self.atoms = ads_atoms
        self.bond_indices = bond_indicies
        self.smiles = smiles


class CustomBulk(Bulk):
    def __init__(self, bulk_atoms):
        self.bulk_atoms = bulk_atoms


def enumerate_slabs(bulk_dict, max_miller=2):

    bulk_dict = copy.deepcopy(bulk_dict)

    logger = logging.getLogger("distributed.worker")
    logger.info("enumerate_slabs_started: %s" % str(bulk_dict))

    bulk_obj = CustomBulk(bulk_dict["bulk_atoms"])

    surfaces = compute.enumerate_surfaces_for_saving(
        bulk_dict["bulk_atoms"], max_miller=2
    )
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
            }
        )
        surface_list.append(surface_result)

    logger.info("enumerate_slabs_finished: %s" % str(bulk_dict))

    return surface_list


def enumerate_adslabs(surface_ads_combo):

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

    # The adslab_result object takes up a ton of memory because they are all
    # copies of the same atoms object. To reduce memory use, let's just store a
    # shallow copy of the atoms for each, and update the positions. Note that this means
    # all other atoms properties are shared between copies, so be careful if you need to
    # modify things!
    adslab_result_shallow_copy = []
    for atoms in adslab_result:
        atoms_copy = copy.copy(adslab_result[0])
        atoms_copy.positions = atoms.positions
        adslab_result_shallow_copy.append(atoms_copy)

    return adslab_result_shallow_copy


def convert_adslabs_to_graphs(adslab_result, max_neighbors=50, cutoff=6):

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
    surface_dict, ads_dict = copy.deepcopy(surface_adsorbate_combo)

    # Prep the new adslab result from the adsorbate and surface info dicts
    adslab_result = {}
    adslab_result.update(surface_dict)
    adslab_result.update(ads_dict)

    return adslab_result
