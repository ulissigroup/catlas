import pickle
import numpy as np
from ocdata import precompute_sample_structures as compute
from ocdata.surfaces import Surface
from ocdata.combined import Combined
from ocdata.adsorbates import Adsorbate
from ocdata.bulk_obj import Bulk


class CustomAdsorbate(Adsorbate):
    def __init__(self, ads_atoms, bond_indicies, smiles):
        self.atoms = ads_atoms
        self.bond_indices = bond_indicies
        self.smiles = smiles


class CustomBulk(Bulk):
    def __init__(self, bulk_atoms):
        self.bulk_atoms = bulk_atoms


def enumerate_slabs(bulk_atoms, max_miller=2):
    bulk_obj = CustomBulk(bulk_atoms)
    surfaces = compute.enumerate_surfaces_for_saving(bulk_atoms, max_miller)
    surface_list = []
    for surface in surfaces:
        surface_struct, millers, shift, top = surface
        surface_object = Surface(bulk_obj, surface, np.nan, np.nan)
        surface_list.append(
            {
                "slab_surface_object": surface_object,
                "slab_millers": millers,
                "slab_max_miller_index": np.max(millers),
                "slab_shift": shift,
                "slab_top": top,
                "slab_natoms": len(surface_object.surface_atoms),
            }
        )
    return surface_list


def enumerate_adslabs(row):
    surface_obj = row.slab_surface_object
    adsorbate_atoms = row.adsorbate_atoms
    bond_indices = row.adsorbate_bond_indices
    smiles = row.adsorbate_smiles
    adsorbate_obj = CustomAdsorbate(adsorbate_atoms, bond_indices, smiles)
    combo_obj = Combined(adsorbate_obj, surface_obj, enumerate_all_configs=True)
    adslabs_list = combo_obj.adsorbed_surface_atoms
    return adslabs_list
