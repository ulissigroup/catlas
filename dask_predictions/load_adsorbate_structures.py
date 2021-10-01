import pickle
import os.path

dir_path = os.path.dirname(os.path.realpath(__file__))


def load_ocdata_adsorbates():
    with open(dir_path + "/adsorbate_structures/adsorbates.pkl", "rb") as fhandle:
        adsorbates = pickle.load(fhandle)

        adsorbate_list = []
        for index in adsorbates:
            atoms, smiles, bond_indices = adsorbates[index]
            adsorbate_list.append(
                {
<<<<<<< HEAD
                    "adsorbate_atoms": atoms,
                    "adsorbate_smiles": smiles,
                    "adsorbate_bond_indices": bond_indices,
                    "adsorbate_data_source": "ocdata_adsorbates",
=======
                    "adsorbate.atoms": atoms,
                    "adsorbate.smiles": smiles,
                    "adsorbate.bond_indices": bond_indices,
                    "adsorbate.data_source": "ocdata_adsorbates",
>>>>>>> 40fd8ba05eeacb3b2f731259b6c8e7b4354bfb72
                }
            )

        return adsorbate_list
