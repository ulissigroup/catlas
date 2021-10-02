from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import copy
from ocdata.combined import Combined
from ase.optimize import LBFGS

# Module calculator to be shared
calc = None


def direct_energy_prediction(adslabs_list, config_path, checkpoint_path):

    global calc

    if calc is None:
        calc = OCPCalculator(config_path, checkpoint=checkpoint_path)

    predictions_list = []

    for adslab in row.adslabs_list:
        adslab = adslab.copy()
        adslab.set_calculator(calc)
        predictions_list.append(adslab.get_potential_energy())

    return predictions_list


def relaxation_energy_prediction(adslabs_list, config_path, checkpoint_path):

    global calc

    if calc is None:
        calc = OCPCalculator(config_path, checkpoint=checkpoint_path)

    predictions_list = []

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
        predictions_list.append(adslab.get_potential_energy())

    return predictions_list
