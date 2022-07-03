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
import catlas.cache_utils


class CustomAdsorbate(Adsorbate):
    def __init__(self, ads_atoms, bond_indicies, smiles):
        self.atoms = ads_atoms
        self.bond_indices = bond_indicies
        self.smiles = smiles


class CustomBulk(Bulk):
    def __init__(self, bulk_atoms):
        self.bulk_atoms = bulk_atoms


def enumerate_slabs(bulk_dict, max_miller=2):

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
    surface_dict, ads_dict = surface_ads_combo

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

    adslab_result = copy.deepcopy(combo_obj.constrained_adsorbed_surfaces)

    return adslab_result


def convert_adslabs_to_graphs(adslab_result, max_neighbors=50, cutoff=6):

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

    return copy.deepcopy(graph_dict)


def merge_surface_adsorbate_combo(surface_adsorbate_combo):
    surface_dict, ads_dict = surface_adsorbate_combo

    # Prep the new adslab result from the adsorbate and surface info dicts
    adslab_result = {}
    adslab_result.update(copy.deepcopy(surface_dict))
    adslab_result.update(copy.deepcopy(ads_dict))

    return adslab_result
