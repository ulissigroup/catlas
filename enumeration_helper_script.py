import pickle
import numpy as np
from ocdata import precompute_sample_structures as compute
from ocdata.base_atoms.pkls import BULK_PKL, ADSORBATE_PKL
from ocdata.surfaces import Surface
from ocdata.combined import Combined


bulk_db = pickle.load(open('/home/jovyan/shared-scratch/Brook/bulk_object_lookup_dict.pkl', 'rb'))
 

def enumerate_surface_wrap(bulk):
    bulk_atoms, mpid = bulk
    bulk_obj = bulk_db[mpid]
    surfaces = compute.enumerate_surfaces_for_saving(bulk_atoms, max_miller=2)
    surface_list = []
    for surface in surfaces:
        surface_list.append([mpid, bulk_obj, surface])
    return surface_list

def grab_surf_obj(surface_list):
    mpid, bulk_obj, surface = surface_list
    surface_struct, millers, shift, top = surface
    surface_object = Surface(bulk_obj, surface, np.nan, np.nan)
    surface_info = [surface_object, mpid, millers, shift, top]
    return surface_info

def enumerate_adslabs_wrap(surface_adsorbate_combo):
    surface_info_object, adsorbate_obj = surface_adsorbate_combo
    surface_obj, mpid, miller, shift, top = surface_info_object
    adslabs = Combined(adsorbate_obj, surface_obj, enumerate_all_configs=True)
    adslab_list = []
    adsorbate = adsorbate_obj.smiles
    for adslab in adslabs.adsorbed_surface_atoms:
        adslab_list.append([mpid, (miller, shift, top), adsorbate, adslab])
    return adslab_list


