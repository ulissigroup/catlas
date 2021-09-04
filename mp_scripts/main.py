import pickle
import numpy as np
import yaml
from ase.optimize import LBFGS
import multiprocessing as mp
import os
import argparse

from ocdata import precompute_sample_structures as compute
from ocdata.base_atoms.pkls import BULK_PKL, ADSORBATE_PKL
from ocdata.surfaces import Surface
from ocdata.combined import Combined
from ocdata.adsorbates import Adsorbate

from ocpmodels.common.relaxation.ase_utils import OCPCalculator


# Specify Paths
config_path = '/home/jovyan/repos/ocp-modeling/configs/s2ef/2M/gemnet/gemnet-dT.yml'
checkpoint_path = '/home/jovyan/shared-scratch/Brook/gemnet_t_direct_h512_all.pt'

# Initialize the calculator
calc = OCPCalculator(config_path, checkpoint=checkpoint_path)

def predict_E(args):
    surface_adsorbate_combo, maxsize = args
    surface_info_object, adsorbate_obj = surface_adsorbate_combo
    surface_obj, mpid, miller, shift, top = surface_info_object
    adslabs = Combined(adsorbate_obj, surface_obj, enumerate_all_configs=True)
    adsorbate = adsorbate_obj.smiles
    predictions_list = []
    # Create a copy of the adslab so that python doesnt change the calculator of the adslab input
    adslabs_list = adslabs.adsorbed_surface_atoms
    size = len(adslabs_list[0].get_tags())
    if size <= maxsize:
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

def create_inputs():

    bulk_db = pickle.load(open('/home/jovyan/shared-scratch/Brook/bulk_object_lookup_dict.pkl', 'rb'))

    with open(BULK_PKL, 'rb') as f:
        inv_index = pickle.load(f)
    bulks = []
    mpids_to_exclude = []
    mpid_list = ['mp-101','mp-126', 'mp-2']
    # Grab the desired bulk atoms object
    num_of_els = [1,2,3]
    for k in num_of_els:
        for itm in inv_index[k]:
            if itm[1] not in mpids_to_exclude and itm[1] in mpid_list:
                 bulks.append(itm)

    adsorbates_smile_list = ['*OH']

    # Open pkl of adsorbates and grab info for the ones in the smile list         
    with open(ADSORBATE_PKL, 'rb') as f:
        inv_index = pickle.load(f)
    ads_idx = [key for key in inv_index if inv_index[key][1] in adsorbates_smile_list]
    adsorbates_obj = [Adsorbate(ADSORBATE_PKL, specified_index = idx) for idx in ads_idx]

    list_of_slabs, predict_E_inputs = [], []
    maxsize = 200

    for bulk in bulks:
        slabs = enumerate_surface_wrap(bulk)
        list_of_slabs.extend(slabs)
    for ads in adsorbates_obj:
        for slab in list_of_slabs:
            predict_E_inputs.append(((slab, ads), maxsize))

    return predict_E_inputs

def run_predictions(predict_E_inputs, idx, n_procs, out_path="."):
    pool = mp.pool(processes=n_procs)
    predictions = pool.map(predict_E, predict_E_inputs)
    with open(os.path.join(out_path, f"predictions_{idx}.pkl"), "wb") as f:
        pickle.dump(f, predictions)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="No. of total jobs created"
    )
    parser.add_argument(
        "--job-idx",
        type=int,
        default=0,
        help="Index of the job"
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=1,
        help="No. of workers used per job"
    )
    parser.add_argument(
        "--out-path",
        default=".",
        help="Directory to save predictions"
    )
    return parser

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()

    input_args = create_inputs()

    # chunk the generated input args into different chunks
    chunked_inputs = np.array_split(input_args, args.num_jobs)

    run_predictions(chunked_inputs[args.job_idx], args.job_idx, args.num_procs, args.out_path)
    command = f"name='job-{idx}' cpu_request={n_procs} cpu_limit={n_procs} bash_command='python run.py '"
