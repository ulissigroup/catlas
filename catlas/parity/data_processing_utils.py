from ase.io.trajectory import Trajectory
import numpy as np
from numpy import load
import pandas as pd
from ase.io.trajectory import Trajectory
from ocdata.flag_anomaly import DetectTrajAnomaly
import re

class ProcessValNPZ:
    def __init__(self, npz_path: str, dft_df_path: str):
        """
        Initialize a class to handle generation of direct approach val dfs.

        Args:
            npz_path: a relative path (from the main catlas directory)
                to the npz path containing inference on val
            dft_df_path: a relative path (from the main catlas directory)
                to df with meta data and ground truth DFT vals
        """
        self.npz_path = npz_path
        self.dft_df_path = dft_df_path
    
    @staticmethod
    def _get_predicted_E(row, ML_data):
        """Finds the corresponding ML energy for a given DFT calculation"""
        random_id = row.random_id
        distribution = row.distribution
        id_now = re.findall(r"([0-9]+)", random_id)[0]
        if distribution == "id":
            idx = np.where(ML_data["id_ids"] == id_now)
            energy = ML_data["id_energy"][idx]
        elif distribution == "ood_ads":
            idx = np.where(ML_data["ood_ads_ids"] == id_now)
            energy = ML_data["ood_ads_energy"][idx]
        elif distribution == "ood_cat":
            idx = np.where(ML_data["ood_cat_ids"] == id_now)
            energy = ML_data["ood_cat_energy"][idx]
        elif distribution == "ood":
            idx = np.where(ML_data["ood_both_ids"] == id_now)
            energy = ML_data["ood_both_energy"][idx]
        return energy[0]
    
    @staticmethod
    def _get_bulk_elements_and_num(stoichiometry):
        return list(stoichiometry.keys()), len(list(stoichiometry.keys()))

    def process_data(self):
        """Creates the primary dataframe for use in analysis"""
        
        model_id = self.npz_path.split("/")[-1]
        model_id = model_id.split(".")[0]

        df_path = "catlas/parity/df_pkls/" + model_id + ".pkl"

        # Open files
        ML_data = dict(load(self.npz_path))
        dft_df = pd.read_pickle(self.dft_df_path)

        # Get ML energies
        dft_df["ML_energy"] = dft_df.apply(self._get_predicted_E, args=(ML_data,), axis=1)
        dft_df["bulk_elements"], dft_df["bulk_nelements"] = zip(*dft_df.stoichiometry.apply(self._get_bulk_elements_and_num))
        dft_df.rename(columns={"energy dE [eV]": "DFT_energy"}, inplace=True)

        # Pickle results for future use
        dft_df.to_pickle(df_path)
    
class ProcessValTraj:
    def __init__(self, traj_path):
        """
        Initialize class to handle trajectory analysis
        
        Args: 
            traj_path: the path to the trajectory to be considered
        """

    def _get_status(self, traj, frame):
        try:
            status = {}
            detector = DetectTrajAnomaly(traj[0], traj[frame], traj[0].get_tags())
            status['dissociation'] = detector.is_adsorbate_dissociated()
            status['desorption'] = detector.is_adsorbate_desorbed(neighbor_thres=3)
            status['reconstruction'] = detector.is_surface_reconstructed(slab_movement_thres=1)
            return all([~status['dissociation'], ~status['desorption'], ~status['reconstruction']])

        except:
            return False

    def _get_energies(self, traj):
        try:
            energies = []
            for frame in traj:
                energy = frame.get_potential_energy()
                energies.append(energy)
            return energies
        except:
            return [np.nan]* len(traj)

    def get_result(self):
        name = file.split("/")[-1].split(".")[0]
        random_num = name.split('/')[-1]
        random_id = "random"+str(random_num)
        try:
            traj = Trajectory(self.traj_path)
            status = self._get_status(traj, -1)
            energy = self._get_energies(traj)
            now = {"random_id": random_id, 'energies' :energy, 'status': status}
            return now

        except:
            return {"random_id": random_id, 'energies' :"failed", 'status': "failed"}
    
