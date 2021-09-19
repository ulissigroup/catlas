from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import copy
from ocdata.combined import Combined

# specify paths
config_path = '/home/jovyan/ocp/configs/s2ef/2M/gemnet/gemnet-dT.yml'
checkpoint_path = '/home/jovyan/shared-scratch/Brook/gemnet_t_direct_h512_all.pt'

#Module calculator to be chared
calc = None

def enumerate_adslabs(surface_adsorbate_combo):

    surface_info_object, adsorbate_obj = surface_adsorbate_combo
    surface_obj, mpid, miller, shift, top = surface_info_object
    adslabs = Combined(adsorbate_obj, surface_obj, enumerate_all_configs=True)
    adsorbate = adsorbate_obj.smiles

    return [surface_adsorbate_combo, adslabs]

def direct_energy_prediction(enumerated_adslabs, direct):

    if calc is None:
        calc = OCPCalculator(config_path, checkpoint=checkpoint_path)

    surface_info_object, adslabs = enumerated_adslabs

    predictions_list = []
    adslabs_list = adslabs.adsorbed_surface_atoms

    if len(adslabs_list) == 0:
        surface_info = [miller,shift,top]
        pred = [mpid,surface_info, adsorbate, 'no valid placements']

    else:
        size = len(adslabs_list[0].get_tags())

        for adslab in adslabs.adsorbed_surface_atoms:
            adslab.set_calculator(calc)
            energy_now = adslab.get_potential_energy()
            predictions_list.append(energy_now)

        surface_info = [miller,shift,top]
        pred = [mpid,surface_info, adsorbate, predictions_list]

    return pred
