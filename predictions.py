import pickle 
import numpy as np
import random
import dask
import dask.bag as db
import functools
import yaml
from dask_kubernetes import KubeCluster
from joblib import Memory
from dask.distributed import Client
from calculator_upload_empty import predict_E
from ocpmodels.models.dimenet_plus_plus import DimeNetPlusPlusWrap
import os
from enumeration_helper_script import enumerate_surface_wrap
from tqdm import tqdm
from ocdata.adsorbates import Adsorbate
from ocdata.bulk_obj import Bulk
from ocdata.base_atoms.pkls import BULK_PKL, ADSORBATE_PKL
from dask import delayed
# Initiate some helpful accessories
#------------------------------------------------------------------------------
name_to_num_dict = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N',
                    8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si',
                    15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc',
                    22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni',
                    29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 
                    36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo',
                    43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In',
                    50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 
                    57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 
                    64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
                    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 
                    78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po',
                    85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa',
                    92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf',
                    99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf',
                    105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 
                    111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv',
                    117: 'Ts', 118: 'Og'}

transition_metal1 = list(range(21,31))
transition_metal2 = list(range(39, 49))
transition_metal3 = list(range(72, 80))
transition_metals = [*transition_metal1, *transition_metal2, *transition_metal3]
# you can set elements_to_include = transition_metals
# ----------------------------------------------------------------------------

# INPUTS!!
# ----------------------------------------------------------------------------
# Bulk inputs
direct = False # boolean True = direct method, False = relaxations
mpids_to_slabs = True # True = use mpid_list to generate aslabs, False = use element list and num_of_els
elements_to_include = [29]# list of desired atomic numbers
mpid_list = ['mp-101']#, 'mp-2', 'mp-30', 'mp-74', 'mp-126', 'mp-101', 'mp-23', 'mp-124', 'mp-102']
num_of_els = [1] # [1,2] = unary and binary
mpids_to_exclude = ['mp-672234', 'mp-632250', 'mp-754514', 'mp-570747', 'mp-12103', 'mp-25', 
                    'mp-672233', 'mp-568584', 'mp-154', 'mp-999498', 'mp-14', 'mp-96', 'mp-1080711', 
                    'mp-1008394', 'mp-22848', 'mp-160', 'mp-1198724'] #these are odd and cause failures

# Adsorbate inputs
adsorbates_smile_list = ['*OH']

# File paths
worker_spec_path = './dask-predictions/worker-spec-relax.yml'
your_shared_scratch_path = '/home/jovyan/shared-scratch/zulissi' # this will be used to create a cache dir in your folder

# Other
num_workers = 4 # should be scaled with workload. Use 1 to troubleshoot. 10 is a good place to start
max_size = 100 # This is the maximum number of atoms you will allow in your slabs. Any greater will return size = Number rather than a prediction.
# -----------------------------------------------------------------------------

# Pre-work / organization
# -----------------------------------------------------------------------------
# convert el nums to names
elnames_to_include = [name_to_num_dict[el] for el in elements_to_include] 

# -----------------------------------------------------------------------------

# Create the workers
# -----------------------------------------------------------------------------
# Set the up a kube dask cluster 
cluster = KubeCluster(worker_spec_path, deploy_mode='local')
cluster.adapt(minimum=1, maximum = num_workers, interval = '30000 ms')
client = Client(cluster)

# Set up the cache directory - this will also be the local directory on each worker that will store cached results
location_preds = your_shared_scratch_path + '/cachedir_predictions_AIChE1'
location_slabs = your_shared_scratch_path + '/cachedir_slabs_AIChEpost'
memory_preds = Memory(location_preds,verbose=1)
memory_slabs = Memory(location_slabs,verbose=1)

def load_bulk_pkl():
    with open(BULK_PKL,'rb') as f:
        bulks = pickle.load(f)
        return bulks[1] #+bulks[2]+bulks[3]

def load_all_ads():
    return [Adsorbate(ADSORBATE_PKL, specified_index = idx) \
                      for idx in range(82)]

def filter_bulk(bulk,
                max_elements = 1,
                mpids_to_exclude = ['mp-672234', 'mp-632250', 'mp-754514', 'mp-570747', 'mp-12103', 'mp-25', 
                                    'mp-672233', 'mp-568584', 'mp-154', 'mp-999498', 'mp-14', 'mp-96', 
                                    'mp-1080711', 'mp-1008394', 'mp-22848', 'mp-160', 'mp-1198724'],
               elements_to_include = set(['Pt'])):
    atoms, mpid = bulk
    unique_elements = np.unique(atoms.get_chemical_symbols()) # Grab the chemical symbol of all
    
    return len(unique_elements)<=max_elements \
            and (mpid not in mpids_to_exclude) \
            and set(unique_elements).issubset(elements_to_include)
        
def filter_adsorbates_smile_list(adsorbate, adsorbates_smile = ['*C']):
    return adsorbate.smiles in adsorbates_smile

#create a dask bag of adsorbates
adsorbates_delayed = dask.delayed(load_all_ads)()
adsorbate_bag = db.from_delayed([adsorbates_delayed])

#create a dask bag of bulks
bulks_delayed = dask.delayed(load_bulk_pkl)()
bulk_bag = db.from_delayed([bulks_delayed])
bulk_bag = bulk_bag.filter(filter_bulk)

# generate the adslab mappings
surfaces_bag = bulk_bag.map(memory_slabs.cache(enumerate_surface_wrap)).flatten()
surface_ads_combos = surfaces_bag.product(adsorbate_bag).repartition(npartitions=num_workers*20).compute()

#Load surfaces into dask bag
surfaces = db.from_sequence(surface_ads_combos)

# generate prediction mappings
predictions_bag = surfaces.map(memory_preds.cache(predict_E), max_size, direct)

# execute operations (go to all work)
predictions = predictions_bag.compute() # change to .compute() to push to local RAM




