import pickle
import numpy as np
import dask
import dask.bag as db
import functools
import yaml
from dask_kubernetes import KubeCluster
from joblib import Memory
from dask.distributed import Client, progress, wait
from calculator_upload_empty import (
    enumerate_adslabs,
    direct_energy_prediction,
    relaxation_energy_prediction,
)

from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram
from pymatgen.ext.matproj import MPRester

from ocpmodels.models.dimenet_plus_plus import DimeNetPlusPlusWrap
import os
from enumerate_surfaces import enumerate_surfaces
from tqdm import tqdm
from ocdata.adsorbates import Adsorbate
from ocdata.bulk_obj import Bulk
from ocdata.base_atoms.pkls import BULK_PKL, ADSORBATE_PKL
from dask import delayed
import dask
from dask_utils import bag_split_individual_partitions
import sys
import time

# Load inputs and define global vars
if __name__ == "__main__":
    inputs_path = sys.argv[1]

with open(inputs_path) as f:
    inputs = yaml.safe_load(f)

## general
adsorbate_smiles = inputs["adsorbate_smile_list"]
num_workers = inputs["num_workers"]

output_options = inputs["output_options"]

## filters
size_filter = inputs["filter_by_object_size"]
max_size = size_filter["max_size"]

mpid_filter = inputs["filter_by_mpids"]
mpids_to_include = mpid_filter["mpid_list"]

element_filter = inputs["filter_by_elements"]
elements_to_include = element_filter["element_list"]

num_elements_filter = inputs["filter_by_number_elements"]
num_elements = num_elements_filter["number_of_els"]

pourbaix_filter = inputs['filter_by_Pourbaix_stability']
pH_allowed = pourbaix_filter['pH']
pH_step = pourbaix_filter['pH_step']
pH_range = np.arange(pH_allowed[0], (float(pH_allowed[1]) + float(pH_step)), pH_step)
AP_allowed = pourbaix_filter['applied_potential']
AP_step = pourbaix_filter['applied_potential_step']
AP_range = np.arange(AP_allowed[0], (float(AP_allowed[1]) + float(AP_step)), AP_step)
decomp_max = pourbaix_filter['max_decomposition_E']

## paths
checkpoint_path = inputs["checkpoint_path"]
config_path = inputs["config_path"]
worker_spec_path = inputs["worker_spec_path"]







with open('/home/jovyan/ocp/dask-predictions/mpid_to_most_recent_mpid.pkl', 'rb') as f:
    old_mpid_to_new = pickle.load(f)


def load_bulk_pkl():
    with open(BULK_PKL, "rb") as f:
        bulks = pickle.load(f)
        return bulks[1] + bulks[2] + bulks[3]
    
def pourbaix_grabber(mpid):
    with MPRester('MGOdX3P4nI18eKvE') as mpr:
        composition = mpr.get_entries(mpid)[0].composition
    comp_dict = {str(key): value for key, value in composition.items()
                         if str(key) not in ['H','O']}
    entries = mpr.get_pourbaix_entries(list(comp_dict.keys()))
    entry = [entry for entry in entries if entry.entry_id == mpid][0]
    pbx = PourbaixDiagram(entries, comp_dict=comp_dict, filter_solids=False)
    return pbx, entry


def load_all_ads():
    return [Adsorbate(ADSORBATE_PKL, specified_index=idx) for idx in range(82)]


def filter_bulk_by_number_elements(bulk, num_elements=num_elements):
    atoms, mpid = bulk
    unique_elements = np.unique(atoms.get_chemical_symbols())
    return len(unique_elements) in num_elements


def filter_bulk_by_elements(bulk, elements_to_include=elements_to_include):
    atoms, mpid = bulk
    unique_elements = np.unique(atoms.get_chemical_symbols())
    return set(unique_elements).issubset(elements_to_include)


def filter_bulk_by_mpids(bulk, mpids_to_include=mpids_to_include):
    atoms, mpid = bulk
    return mpid in mpids_to_include


def filter_out_bad_mpids(
    bulk,
    mpids_to_exclude=[
        "mp-672234",
        "mp-632250",
        "mp-754514",
        "mp-570747",
        "mp-12103",
        "mp-25",
        "mp-672233",
        "mp-568584",
        "mp-154",
        "mp-999498",
        "mp-14",
        "mp-96",
        "mp-1080711",
        "mp-1008394",
        "mp-22848",
        "mp-160",
        "mp-1198724",
    ],
):
    atoms, mpid = bulk
    return mpid not in mpids_to_exclude


def filter_adsorbates_smile_list(adsorbate, adsorbate_smile_list=adsorbate_smiles):
    return adsorbate.smiles in adsorbate_smile_list


def filter_surfaces_by_size(surface, max_size=max_size):
    surface_object, mpid, millers, shift, top = surface
    return len(surface_object.surface_atoms) <= max_size

def filter_bulk_by_pourbaix(bulk, old_mpid_to_new = old_mpid_to_new, pH_range = pH_range,
                            AP_range = AP_range, decomp_max = decomp_max):
    atoms, mpid = bulk
    pbx_get = pourbaix_memory.cache(pourbaix_grabber)
    mpid_new = old_mpid_to_new[mpid]
    pbx, entry = pbx_get(mpid_new)
    for pH in pH_range:
        for AP in AP_range:
            decomp_energy = pbx.get_decomposition_energy(entry, pH, AP)
            if decomp_energy <= decomp_max:
                return True
    return False
            
    


def run_filter_bulks_enumerate_adslabs():

    # create a dask bag of bulks
    bulks_delayed = dask.delayed(load_bulk_pkl)()
    bulk_bag = db.from_delayed([bulks_delayed]).repartition(npartitions=num_workers)

    # Apply mandatory bulk filters
    filtered_bulk_bag = bulk_bag.filter(filter_out_bad_mpids)

    # Apply optional bulk filters
    if num_elements_filter["enable"]:
        filtered_bulk_bag = filtered_bulk_bag.filter(filter_bulk_by_number_elements)

    if element_filter["enable"]:
        filtered_bulk_bag = filtered_bulk_bag.filter(filter_bulk_by_elements)

    if mpid_filter["enable"]:
        filtered_bulk_bag = filtered_bulk_bag.filter(filter_bulk_by_mpids)
        
    if pourbaix_filter['enable']:
        filtered_bulk_bag = filtered_bulk_bag.filter(filter_bulk_by_pourbaix)

    # Repartition
    filtered_bulk_bag = bag_split_individual_partitions(filtered_bulk_bag)

    # generate the adslab mappings
    surfaces_bag = filtered_bulk_bag.map(memory.cache(enumerate_surfaces)).flatten()

    # filter surfaces to find those with less than the desired number of atoms
    if size_filter["enable"]:
        surfaces_bag = surfaces_bag.filter(filter_surfaces_by_size)

    # create a dask bag of adsorbates
    adsorbates_delayed = dask.delayed(load_all_ads)()
    adsorbate_bag = db.from_delayed([adsorbates_delayed])
    adsorbate_bag_filtered = adsorbate_bag.filter(filter_adsorbates_smile_list)

    # generate surface/adsorbate combos
    surface_ads_combos = surfaces_bag.product(adsorbate_bag_filtered)
    surface_ads_combos = bag_split_individual_partitions(surface_ads_combos)

    enumerated_adslabs = surface_ads_combos.map(memory.cache(enumerate_adslabs))

    return enumerated_adslabs


def run_predictions(enumerated_adslabs, mode):

    # generate prediction mappings
    if mode:
        predictions = enumerated_adslabs.map(
            memory.cache(direct_energy_prediction), config_path, checkpoint_path
        )
    else:
        predictions = enumerated_adslabs.map(
            memory.cache(relaxation_energy_prediction), config_path, checkpoint_path
        )

    return predictions


cache_location = inputs["shared_scratch_path"] + "/cachedir"
memory = Memory(cache_location, verbose=1)
pourbaix_memory = Memory('/home/jovyan/shared-scratch/Inference/pourbaix_cachedir', verbose=1)

cluster = KubeCluster(worker_spec_path, deploy_mode="local")
cluster.adapt(minimum=1, maximum=num_workers, interval="30000 ms")
client = Client(cluster)

enumerated_adslabs = run_filter_bulks_enumerate_adslabs()
predictions = run_predictions(enumerated_adslabs, inputs["direct_prediction_mode"])

if output_options['pickle']:
    final_predictions = predictions.compute()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = inputs["shared_scratch_path"]+ '/' + timestr + output_options['optional_additional_path_str'] + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(final_predictions, f)
    
else:
    final_predictions = predictions.persist()

