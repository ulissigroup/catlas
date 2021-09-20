import pickle
import numpy as np
import dask
import dask.bag as db
import functools
import yaml
from dask_kubernetes import KubeCluster
from joblib import Memory
from dask.distributed import Client, progress, wait
from calculator_upload_empty import enumerate_adslabs, direct_energy_prediction

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


def load_bulk_pkl():
    with open(BULK_PKL,'rb') as f:
        bulks = pickle.load(f)
        return bulks[1]+bulks[2]+bulks[3]

def load_all_ads():
    return [Adsorbate(ADSORBATE_PKL, specified_index = idx) \
                     for idx in range(82)]

def filter_bulk_by_number_elements(bulk, max_elements=2):
    atoms, mpid = bulk
    unique_elements = np.unique(atoms.get_chemical_symbols())
    return len(unique_elements)<=max_elements

def filter_bulk_by_elements(bulk,  elements_to_include = set(['Pt', 'Cu', 'Pd', 'Ag','Au', 'Ni', 'Fe', 'Rh', 'Ru', 'Sn', 'Zn', 'Co', 'Cd', 'Mn', 'Ir'])):
    atoms, mpid = bulk
    unique_elements = np.unique(atoms.get_chemical_symbols())
    return set(unique_elements).issubset(elements_to_include)

def filter_out_bad_mpids(bulk, mpids_to_exclude = ['mp-672234', 'mp-632250', 'mp-754514', 'mp-570747', 'mp-12103', 'mp-25',
                                    'mp-672233', 'mp-568584', 'mp-154', 'mp-999498', 'mp-14', 'mp-96',
                                    'mp-1080711', 'mp-1008394', 'mp-22848', 'mp-160', 'mp-1198724']):
    atoms, mpid = bulk
    return mpid not in mpids_to_exclude

def filter_adsorbates_smile_list(adsorbate, adsorbates_smile = ['*C', '*CO']):
    return adsorbate.smiles in adsorbates_smile

def filter_surfaces_by_size(surface, max_size=80):
    surface_object, mpid, millers, shift, top = surface
    return len(surface_object.surface_atoms)<max_size

def run_filter_bulks_enumerate_adslabs():

    # create a dask bag of bulks
    bulks_delayed = dask.delayed(load_bulk_pkl)()
    bulk_bag = db.from_delayed([bulks_delayed]).repartition(npartitions=num_workers)

    # Filter the bulks
    filtered_bulk_bag = bulk_bag.filter(filter_bulk_by_number_elements)
    filtered_bulk_bag = filtered_bulk_bag.filter(filter_bulk_by_elements)
    filtered_bulk_bag = filtered_bulk_bag.filter(filter_out_bad_mpids)
    filtered_bulk_bag = bag_split_individual_partitions(filtered_bulk_bag)

    # generate the adslab mappings
    surfaces_bag = filtered_bulk_bag.map(memory.cache(enumerate_surface_wrap)).flatten()

    # filter surfaces to find those with less than the desired number of atoms
    surfaces_bag_filtered = surfaces_bag.filter(filter_surfaces_by_size)

    # create a dask bag of adsorbates
    adsorbates_delayed = dask.delayed(load_all_ads)()
    adsorbate_bag = db.from_delayed([adsorbates_delayed])
    adsorbate_bag_filtered = adsorbate_bag.filter(filter_adsorbates_smile_list)

    # generate surface/adsorbate combos
    surface_ads_combos = surfaces_bag_filtered.product(adsorbate_bag_filtered)
    surface_ads_combos = bag_split_individual_partitions(surface_ads_combos)

    enumerated_adslabs = surface_ads_combos.map(memory.cache(enumerate_adslabs))

    return enumerated_adslabs

def run_predictions(enumerated_adslabs):

    # generate prediction mappings
    predictions = enumerated_adslabs.map(memory.cache(direct_energy_prediction))
    return predictions

your_shared_scratch_path = '/home/jovyan/shared-scratch/zulissi' # this will be used to create a cache dir in your folder
cache_location = your_shared_scratch_path + '/cachedir'
memory = Memory(cache_location,verbose=1)

# Other
num_workers = 100 # should be scaled with workload. Use 1 to troubleshoot. 10 is a good place to start

cluster = KubeCluster('./dask-predictions/worker-spec.yml', deploy_mode='local')
cluster.adapt(minimum=2, maximum = num_workers, interval = '30000 ms')
client = Client(cluster)

enumerated_adslabs = run_filter_bulks_enumerate_adslabs().persist()
wait(enumerated_adslabs)

cluster.close()
#client.shutdown()
client.close()

import time
time.sleep(30)

cluster  = KubeCluster('./dask-predictions/worker-spec-gpu.yml', deploy_mode='local')
cluster.adapt(minimum=2, maximum = num_workers, interval = '30000 ms')
client = Client(cluster)

enumerated_adslabs = run_filter_bulks_enumerate_adslabs()
predictions = run_predictions(enumerated_adslabs)
final_predictions = predictions.compute()
#progress(predictions)

