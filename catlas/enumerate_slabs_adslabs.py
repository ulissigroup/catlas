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


def enumerate_slabs(bulk_info, max_miller=2):
    bulk_atoms, mpid, source, ns, functional, ns2, composition = bulk_info
    bulk_obj = CustomBulk(bulk_atoms)
    surfaces = compute.enumerate_surfaces_for_saving(bulk_atoms, max_miller=2)
    surface_list = []
    for surface in surfaces:
        surface_struct, millers, shift, top = surface
        surface_object = Surface(bulk_obj, surface, np.nan, np.nan)
        surface_list.append(
            {
                "slab_surface_object": surface_object,
                "slab_millers": millers,
                "mpid": mpid,
                "source" : source,
                "composition" : composition,
                "bulk_functional": functional,
                "slab_shift": shift,
                "slab_top": top,
                "slab_natoms": len(surface_object.surface_atoms),
            }
        )
    return surface_list


def enumerate_adslabs(surface_ads_combo):
    surface_stuff = surface_ads_combo[0]
    adsorbate_stuff = surface_ads_combo[1]
    surface_obj = surface_stuff['slab_surface_object']
    adsorbate_atoms = adsorbate_stuff['adsorbate_atoms']
    bond_indices = adsorbate_stuff['adsorbate_bond_indices']
    smiles = adsorbate_stuff['adsorbate_smiles']
    adsorbate_obj = CustomAdsorbate(adsorbate_atoms, bond_indices, smiles)
    combo_obj = Combined(adsorbate_obj, surface_obj, enumerate_all_configs=True)
    adslabs_list = combo_obj.adsorbed_surface_atoms
    dict_to_return = {
        'adsorbate_smile' : smiles,
        'mpid': surface_stuff['mpid'],
        'bulk_source': surface_stuff['source'],
        'shift': surface_stuff['slab_shift'],
        'slab_top': surface_stuff['slab_top'],
        'bulk_functional': surface_stuff['bulk_functional'],
        'slab_millers': surface_stuff['slab_millers'],
        'adslab_atoms': adslabs_list
    }
    return dict_to_return
