from ocpmodels.trainers import EnergyTrainer
from ocpmodels import models
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import copy
import yaml

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

def predict_E(adslab):
          
    # Create a copy of the adslab so that python doesnt change the calculator of the adslab input
    adslab = copy.deepcopy(adslab)
    
    # Unpack data
    mpid, surface_info, adsorbate, adslab_atoms = adslab
    
    #Iterate over sites and predict on each
    adslab_atoms.set_calculator(calc)
    energy_now = adslab_atoms.get_potential_energy()
    pred = [mpid,surface_info, adsorbate, energy_now]
    return pred
