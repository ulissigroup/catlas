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
        surface_struct, millers, shift, top = surface
        surface_object = Surface(bulk_obj, surface, np.nan, np.nan)
        surface_info = [surface_object, mpid, millers, shift, top]
        surface_list.append(surface_info)
    return surface_list
