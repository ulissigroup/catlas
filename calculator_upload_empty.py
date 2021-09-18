from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import copy
from ocdata.combined import Combined
from ase.optimize import LBFGS


# specify paths
config_path = '/home/jovyan/ocp/configs/s2ef/2M/gemnet/gemnet-dT.yml'
checkpoint_path = '/home/jovyan/shared-scratch/Brook/gemnet_t_direct_h512_all.pt'
# # File paths
# config_path = '/home/jovyan/ocp/configs/is2re/all/dimenet_plus_plus/dpp.yml'
# checkpoint_path = '/home/jovyan/ocp/data/OC20/pretrained/is2re/dimenetpp_all.pt'

# Initialize the calculator
calc = OCPCalculator(config_path, checkpoint=checkpoint_path)

def predict_E(surface_adsorbate_combo, max_size, direct):
    if direct == True:
        surface_info_object, adsorbate_obj = surface_adsorbate_combo
        surface_obj, mpid, miller, shift, top = surface_info_object
        adslabs = Combined(adsorbate_obj, surface_obj, enumerate_all_configs=True)
        adsorbate = adsorbate_obj.smiles
        predictions_list = []
        adslabs_list = adslabs.adsorbed_surface_atoms
        if len(adslabs_list) == 0:
            surface_info = [miller,shift,top]
            pred = [mpid,surface_info, adsorbate, 'no valid placements']
        else:
            size = len(adslabs_list[0].get_tags())
            if size <= max_size:
            # Create a copy of the adslab so that python doesnt change the calculator of the adslab input
                for adslab in adslabs.adsorbed_surface_atoms:
                    adslab.set_calculator(calc)
                    energy_now = adslab.get_potential_energy()
                    predictions_list.append(energy_now)
            else:
                predictions_list =[f'size = {size}']
            surface_info = [miller,shift,top]
            pred = [mpid,surface_info, adsorbate, predictions_list]
        return pred
    else:
        surface_info_object, adsorbate_obj = surface_adsorbate_combo
        surface_obj, mpid, miller, shift, top = surface_info_object
        adslabs = Combined(adsorbate_obj, surface_obj, enumerate_all_configs=True)
        adsorbate = adsorbate_obj.smiles
        predictions_list = []
        adslabs_list = adslabs.adsorbed_surface_atoms
        if len(adslabs_list) == 0:
            surface_info = [miller,shift,top]
            pred = [mpid,surface_info, adsorbate, 'no valid placements']
        else:
            size = len(adslabs_list[0].get_tags())
            if size <= 1000:
                for adslab in adslabs.adsorbed_surface_atoms:
                    adslab.set_calculator(calc)
                    opt = LBFGS(
                        adslab,
                        maxstep= 0.04,
                        memory= 1,
                        damping= 1.0,
                        alpha= 70.0,
                        trajectory=None,
                    )
                    opt.run(fmax=0.05, steps = 200)
                    energy_now = adslab.get_potential_energy()
                    predictions_list.append(energy_now)
            else:
                predictions_list =[f'size = {size}']
            surface_info = [miller,shift,top]
            pred = [mpid,surface_info, adsorbate, predictions_list]
        return pred
