import pickle
import numpy as np
from ocdata import precompute_sample_structures as compute
from ocdata.surfaces import Surface


# We shouldn't need things like this
# bulk_db = pickle.load(
#    open("/home/jovyan/shared-scratch/Brook/bulk_object_lookup_dict.pkl", "rb")
# )


def enumerate_slabs(bulk):
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


def enumerate_adslabs(surface_adsorbate_combo):
    print(surface_adsorbate_combo)
    surface_info_object, adsorbate_obj = surface_adsorbate_combo
    surface_obj, mpid, miller, shift, top = surface_info_object
    adslabs = Combined(adsorbate_obj, surface_obj, enumerate_all_configs=True)
    adsorbate = adsorbate_obj.smiles

    return [surface_info_object, surface_adsorbate_combo, adslabs]
