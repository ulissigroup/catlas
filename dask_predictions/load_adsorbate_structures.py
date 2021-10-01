import pickle
import os.path

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_ocdata_adsorbates():
    with open(dir_path+'/adsorbate_structures/adsorbates.pkl','rb') as fhandle:
        adsorbates = pickle.load(fhandle)

        adsorbate_list = []
        for atoms, smiles, bond_indices in adsorbates:
            adsorbate_list.append({'adsorbate.atoms': atoms,
                                   'adsorbate.smiles': smiles,
                                   'adsorbate.bond_indices': bond_indices,
                                   'adsorbate.data_source': 'ocdata_adsorbates'})

        return adsorbate_list
