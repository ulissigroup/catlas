from ocpmodels.trainers import EnergyTrainer
from ocpmodels import models
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import copy
import yaml
from ocdata.combined import Combined

# Load the calculator
config_path = '/home/jovyan/shared-scratch/Brook/ocpcalc_config.yml'
with open(config_path) as f:
    config = yaml.safe_load(f)
# Initializa the trainer for the calculator
trainer = EnergyTrainer(
    task=config["task"],
    model=config["model"],
    dataset=config["dataset"],
    optimizer=config["optim"],
    identifier="dpp-predictions",
    run_dir="./",
    is_debug=False,  # if True, do not save checkpoint, logs, or results
    is_vis=False,
    print_every=10,
    seed=0,  # random seed to use
    logger="tensorboard",  # logger of choice (tensorboard and wandb supported)
    # local_rank=0,
    amp=False,  # use PyTorch Automatic Mixed Precision (faster training and less memory usage)
    cpu=True,
)
# Load the pretrained checkpoint
checkpoint_path_dict = config['checkpoint_path']
checkpoint_path = checkpoint_path_dict['path']

trainer.load_pretrained(checkpoint_path, True)
# Initialize the calculator
calc = OCPCalculator(trainer)

def predict_E(surface_adsorbate_combo, max_size):
    surface_info_object, adsorbate_obj = surface_adsorbate_combo
    surface_obj, mpid, miller, shift, top = surface_info_object
    adslabs = Combined(adsorbate_obj, surface_obj, enumerate_all_configs=True)
    adsorbate = adsorbate_obj.smiles
    predictions_list = []
    adslabs_list = adslabs.adsorbed_surface_atoms
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
