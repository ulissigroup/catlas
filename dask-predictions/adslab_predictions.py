from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import copy
from ocdata.combined import Combined
from ase.optimize import LBFGS

# Module calculator to be chared
calc = None


def direct_energy_prediction(enumerated_adslabs, config_path, checkpoint_path):

    global calc

    if calc is None:
        calc = OCPCalculator(config_path, checkpoint=checkpoint_path)

    surface_info_object, surface_adsorbate_combo, adslabs = enumerated_adslabs
    surface_obj, mpid, miller, shift, top = surface_info_object
    surface_info_object, adsorbate_obj = surface_adsorbate_combo
    adsorbate = adsorbate_obj.smiles

    predictions_list = []
    adslabs_list = adslabs.adsorbed_surface_atoms

    if len(adslabs_list) == 0:
        surface_info = [miller, shift, top]
        pred = [mpid, surface_info, adsorbate, "no valid placements"]

    else:

        for adslab in adslabs_list:
            adslab = adslab.copy()
            adslab.set_calculator(calc)
            energy_now = adslab.get_potential_energy()
            predictions_list.append(energy_now)

        surface_info = [miller, shift, top]
        pred = [mpid, surface_info, adsorbate, predictions_list]

    return pred


def relaxation_energy_prediction(enumerated_adslabs, config_path, checkpoint_path):

    global calc

    if calc is None:
        calc = OCPCalculator(config_path, checkpoint=checkpoint_path)

    surface_info_object, surface_adsorbate_combo, adslabs = enumerated_adslabs
    surface_obj, mpid, miller, shift, top = surface_info_object
    surface_info_object, adsorbate_obj = surface_adsorbate_combo
    adsorbate = adsorbate_obj.smiles

    predictions_list = []
    adslabs_list = adslabs.adsorbed_surface_atoms

    if len(adslabs_list) == 0:
        surface_info = [miller, shift, top]
        pred = [mpid, surface_info, adsorbate, "no valid placements"]

    else:

        for adslab in adslabs_list:
            adslab = adslab.copy()
            adslab.set_calculator(calc)
            opt = LBFGS(
                adslab,
                maxstep=0.04,
                memory=1,
                damping=1.0,
                alpha=70.0,
                trajectory=None,
            )
            opt.run(fmax=0.05, steps=200)
            energy_now = adslab.get_potential_energy()
            predictions_list.append(energy_now)

        surface_info = [miller, shift, top]
        pred = [mpid, surface_info, adsorbate, predictions_list]

    return pred
