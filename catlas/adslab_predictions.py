from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import copy
from ocdata.combined import Combined
from ase.optimize import LBFGS

# Module calculator to be shared
direct_calc = None
relax_calc = None


def direct_energy_prediction(adslab_dict, config_path, checkpoint_path, column_name):

    adslab_results = copy.deepcopy(adslab_dict)

    global direct_calc

    if direct_calc is None:
        direct_calc = OCPCalculator(config_path, checkpoint=checkpoint_path)

    predictions_list = []

    for adslab in adslab_dict["adslab_atoms"]:
        adslab = adslab.copy()
        adslab.set_calculator(direct_calc)
        predictions_list.append(adslab.get_potential_energy())

    adslab_results[column_name] = predictions_list
    adslab_results["min_" + column_name] = min(predictions_list)

    return adslab_results


def relaxation_energy_prediction(
    adslabs_dict, config_path, checkpoint_path, column_name
):

    adslab_results = copy.deepcopy(adslab_dict)

    global relax_calc

    if relax_calc is None:
        relax_calc = OCPCalculator(config_path, checkpoint=checkpoint_path)

    predictions_list = []

    for adslab in adslab_dict["adslab_atoms"]:
        adslab = adslab.copy()
        adslab.set_calculator(relax_calc)
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

    adslab_results[column_name] = predictions_list
    adslab_results["min_" + column_name] = min(predictions_list)

    return adslab_results
