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
                    "adsorbate_atoms": atoms,
                    "adsorbate_smiles": smiles,
                    "adsorbate_bond_indices": bond_indices,
                    "adsorbate_data_source": "ocdata_adsorbates",
                }
            )

        return adsorbate_list
