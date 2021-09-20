import pickle
import numpy as np
import dask
import dask.bag as db
import functools
import yaml
from dask_kubernetes import KubeCluster
from joblib import Memory
from dask.distributed import Client, progress, wait
from calculator_upload_empty import enumerate_adslabs, direct_energy_prediction, relaxation_energy_prediction

from ocpmodels.models.dimenet_plus_plus import DimeNetPlusPlusWrap
import os
from enumeration_helper_script import enumerate_surface_wrap
from tqdm import tqdm
from ocdata.adsorbates import Adsorbate
from ocdata.bulk_obj import Bulk
from ocdata.base_atoms.pkls import BULK_PKL, ADSORBATE_PKL
from dask import delayed
import dask
from dask_utils import bag_split_individual_partitions
import sys

# Load inputs and define global vars
if __name__ == '__main__':
    inputs_path = sys.argv[1]
    
with open(inputs_path) as f:
    inputs = yaml.safe_load(f)
    
adsorbate_smiles = inputs['adsorbate_smile_list']
num_workers = inputs['num_workers'] # should be scaled with workload. Use 1 to troubleshoot. 10 is a good place to start
size_filter = inputs['filter_by_object_size']
mpid_filter = inputs['filter_by_mpids']
element_filter = inputs['filter_by_elements']
num_elements_filter = inputs['filter_by_number_elements']
checkpoint_path = inputs['checkpoint_path']
config_path = inputs['config_path']
worker_spec_path = inputs['worker_spec_path']
num_elements = num_elements_filter['number_of_els']
elements_to_include = element_filter['element_list']
mpids_to_include = mpid_filter['mpid_list']
max_size = size_filter['max_size']

def load_bulk_pkl():
    with open(BULK_PKL,'rb') as f:
        bulks = pickle.load(f)
        return bulks[1]+bulks[2]+bulks[3]

def load_all_ads():
    return [Adsorbate(ADSORBATE_PKL, specified_index = idx) \
                     for idx in range(82)]

def filter_bulk_by_number_elements(bulk, num_elements = num_elements):
    atoms, mpid = bulk
    unique_elements = np.unique(atoms.get_chemical_symbols())
    return len(unique_elements) in num_elements

def filter_bulk_by_elements(bulk, elements_to_include = elements_to_include):
    atoms, mpid = bulk
    unique_elements = np.unique(atoms.get_chemical_symbols())
    return set(unique_elements).issubset(elements_to_include)

def filter_bulk_by_mpids(bulk, mpids_to_include = mpids_to_include):
    atoms, mpid = bulk
    return mpid in mpids_to_include

def filter_out_bad_mpids(bulk, mpids_to_exclude = ['mp-672234', 'mp-632250', 'mp-754514', 'mp-570747', 'mp-12103', 'mp-25',
                                    'mp-672233', 'mp-568584', 'mp-154', 'mp-999498', 'mp-14', 'mp-96',
                                    'mp-1080711', 'mp-1008394', 'mp-22848', 'mp-160', 'mp-1198724']):
    atoms, mpid = bulk
    return mpid not in mpids_to_exclude

def filter_adsorbates_smile_list(adsorbate, adsorbate_smile_list = adsorbate_smiles):
    return adsorbate.smiles in adsorbate_smile_list

def filter_surfaces_by_size(surface, max_size = max_size):
    surface_object, mpid, millers, shift, top = surface
    return len(surface_object.surface_atoms)<=max_size

def run_filter_bulks_enumerate_adslabs():

    # create a dask bag of bulks
    bulks_delayed = dask.delayed(load_bulk_pkl)()
    bulk_bag = db.from_delayed([bulks_delayed]).repartition(npartitions=num_workers)

    # Apply mandatory bulk filters
    filtered_bulk_bag = bulk_bag.filter(filter_out_bad_mpids)
    
    # Apply optional bulk filters
    if num_elements_filter['enable']:
        filtered_bulk_bag = filtered_bulk_bag.filter(filter_bulk_by_number_elements)
        
    if element_filter['enable']:
        filtered_bulk_bag = filtered_bulk_bag.filter(filter_bulk_by_elements)
    
    if mpid_filter['enable']:
        filtered_bulk_bag = filtered_bulk_bag.filter(filter_bulk_by_mpids)
    
    # Repartition
    filtered_bulk_bag = bag_split_individual_partitions(filtered_bulk_bag)

    # generate the adslab mappings
    surfaces_bag = filtered_bulk_bag.map(memory.cache(enumerate_surface_wrap)).flatten()


    # filter surfaces to find those with less than the desired number of atoms
    if size_filter['enable']:
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
        predictions = enumerated_adslabs.map(memory.cache(direct_energy_prediction), config_path, checkpoint_path)
    else:
        predictions = enumerated_adslabs.map(memory.cache(relaxation_energy_prediction), config_path, checkpoint_path)
    
    return predictions



cache_location = inputs['shared_scratch_path'] + '/cachedir'
memory = Memory(cache_location,verbose=1)


cluster = KubeCluster(worker_spec_path, deploy_mode='local')
cluster.adapt(minimum=2, maximum = num_workers, interval = '30000 ms')
client = Client(cluster)

enumerated_adslabs = run_filter_bulks_enumerate_adslabs()
predictions = run_predictions(enumerated_adslabs, inputs['direct_prediction_mode'])
final_predictions = predictions.compute()
print(final_predictions[0])


