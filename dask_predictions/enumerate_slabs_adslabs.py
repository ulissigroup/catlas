import pickle
import numpy as np
from ocdata import precompute_sample_structures as compute
from ocdata.surfaces import Surface
from dask_predictions.bulk_object import CustomBulk
from dask_predictions.adsorbate_object import CustomAdsorbate
from ocdata.combined import Combined

def enumerate_slabs(bulk_atoms):
    bulk_obj = CustomBulk(bulk_atoms)
    surfaces = compute.enumerate_surfaces_for_saving(bulk_atoms, max_miller=2)
    surface_list = []
    for surface in surfaces:
        surface_struct, millers, shift, top = surface
        surface_info = (millers, shift, top)
        surface_object = Surface(bulk_obj, surface, np.nan, np.nan)
        surface_list.append([surface_object, surface_info])
    return surface_list


def enumerate_adslabs(row):
    surface_obj = row.surfaces[0]
    adsorbate_atoms = row.adsorbate_atoms
    bond_indices = row.adsorbate_bond_indices
    smiles = row.adsorbate_smiles
    adsorbate_obj = CustomAdsorbate(adsorbate_atoms, bond_indices, smiles)
    combo_obj = Combined(adsorbate_obj, surface_obj, enumerate_all_configs=True)
    adslabs_list = combo_obj.adsorbed_surface_atoms
    return adslabs_list
