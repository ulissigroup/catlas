import pickle
import os.path
import catlas

dir_path = os.path.dirname(os.path.realpath(__file__))


def load_ocdata_adsorbates(adsorbate_path):
    path = "%s/%s" % (
        os.path.join(os.path.dirname(catlas.__file__), os.pardir),
        adsorbate_path,
    )
    with open(path, "rb") as fhandle:
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
