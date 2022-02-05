from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import copy
from ocdata.combined import Combined
from ase.optimize import LBFGS
import numpy as np
from ocpmodels.preprocessing import AtomsToGraphs
import yaml
from ocpmodels.datasets.trajectory_lmdb import data_list_collater
from ase.calculators.singlepoint import SinglePointCalculator
from ocpmodels.common.relaxation import ml_relaxation

from ocpmodels.common.utils import (
    radius_graph_pbc,
    setup_imports,
    setup_logging,
)

from torch.utils.data import DataLoader
import torch
import catlas
import os
from tqdm import tqdm

BOCPP_dict = {}
relax_calc = None


def pop_keys(adslab_dict, keys):
    adslab_dict = copy.deepcopy(adslab_dict)
    for key in keys:
        adslab_dict.pop(key)
    return adslab_dict


from torch.utils.data import Dataset
from ocpmodels.common.registry import registry


class GraphsListDataset(Dataset):
    def __init__(self, graphs_list):
        self.graphs_list = graphs_list

    def __len__(self):
        return len(self.graphs_list)

    def __getitem__(self, idx):
        graph = self.graphs_list[idx]
        return graph


class BatchOCPPredictor:
    def __init__(self, checkpoint, batch_size=8, cpu=False):

        setup_imports()
        setup_logging()

        checkpoint = "%s/%s" % (
            os.path.join(os.path.dirname(catlas.__file__), os.pardir),
            checkpoint,
        )

        config = torch.load(checkpoint, map_location=torch.device("cpu"))["config"]

        # Load the trainer based on the dataset used
        if config["task"]["dataset"] == "trajectory_lmdb":  # S2EF
            config["trainer"] = "forces"
        elif config["task"]["dataset"] == "single_point_lmdb":  # IS2RE
            config["trainer"] = "energy"

        config["model_attributes"]["name"] = config.pop("model")
        config["model"] = config["model_attributes"]

        config["model_attributes"]["otf_graph"] = True

        if "normalizer" not in config:
            del config["dataset"]["src"]
            config["normalizer"] = config["dataset"]

        if "scale_file" in config["model_attributes"]:
            catlas_dir = os.path.dirname(catlas.__file__)
            config["model_attributes"]["scale_file"] = "%s/ocp_checkpoints/%s" % (
                os.path.join(os.path.dirname(catlas.__file__), os.pardir),
                config["model_attributes"]["scale_file"],
            )

        config["checkpoint"] = checkpoint

        # Turn off parallel data loading since this doesn't place nicely
        # with dask threads and dask nanny
        config["optim"]["num_workers"] = 0

        self.config = copy.deepcopy(config)

        self.batch_size = batch_size

        self.trainer = registry.get_trainer_class(self.config.get("trainer", "energy"))(
            task=self.config["task"],
            model=self.config["model"],
            dataset=None,
            normalizer=self.config["normalizer"],
            optimizer=self.config["optim"],
            identifier="",
            slurm=self.config.get("slurm", {}),
            local_rank=self.config.get("local_rank", 0),
            is_debug=self.config.get("is_debug", True),
            cpu=cpu,
        )

        self.device = ["cpu" if cpu else "cuda:0"][0]

        self.predict = self.trainer.predict  # TorchCalc expects a predict method

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def make_dataloader(self, graphs_list):

        # Make a dataset
        graphs_list_dataset = GraphsListDataset(graphs_list)

        # Make a loader
        data_loader = self.trainer.get_dataloader(
            graphs_list_dataset,
            self.trainer.get_sampler(
                graphs_list_dataset, self.batch_size, shuffle=False
            ),
        )

        return data_loader

    def load_checkpoint(self, checkpoint_path):
        """
        Load existing trained model
        Args:
            checkpoint_path: string
                Path to trained model
        """
        try:
            self.trainer.load_checkpoint(checkpoint_path)
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def direct_prediction(self, graphs_list):

        data_loader = self.make_dataloader(graphs_list)

        # Batch inference
        predictions = self.trainer.predict(
            data_loader, per_image=True, disable_tqdm=True
        )

        return predictions["energy"]

    def relaxation_prediction(self, graphs_list):

        if self.device == "cpu":
            torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

        data_loader = self.make_dataloader(graphs_list)
        energy_predictions = []
        position_predictions = []

        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            if i >= self.config["task"].get("num_relaxation_batches", 1e9):
                break

            relaxed_batch = ml_relaxation.ml_relax(
                batch=batch,
                model=self,
                steps=self.config["task"].get("relaxation_steps", 200),
                fmax=self.config["task"].get("relaxation_fmax", 0.0),
                relax_opt={"memory": 100},
                device=self.device,
                transform=None,
            )

            # Grab the predicted energies
            energy_predictions.extend(relaxed_batch.y)

            # Grab the relaxed positions as well
            natoms = relaxed_batch.natoms.tolist()
            positions = torch.split(relaxed_batch.pos, natoms)
            position_predictions += [pos.tolist() for pos in positions]

        return energy_predictions, position_predictions


def energy_prediction(
    adslab_dict,
    adslab_atoms,
    graphs_dict,
    checkpoint_path,
    column_name,
    batch_size=8,
    cpu=False,
):

    global BOCPP_dict

    if (checkpoint_path, batch_size, cpu) not in BOCPP_dict:
        BOCPP_dict[checkpoint_path, batch_size, cpu] = BatchOCPPredictor(
            checkpoint=checkpoint_path,
            batch_size=batch_size,
            cpu=cpu,
        )

    BOCPP = BOCPP_dict[checkpoint_path, batch_size, cpu]

    adslab_results = copy.copy(adslab_dict)

    if BOCPP.config["trainer"] == "forces":
        energy_predictions, position_predictions = BOCPP.relaxation_prediction(
            graphs_dict["adslab_graphs"]
        )
        energy_predictions = np.array([p.cpu().numpy() for p in energy_predictions])

        # Use the relaxed positions to generate relaxed atoms objects
        adslab_atoms = copy.deepcopy(adslab_atoms)
        for atoms, positions in zip(adslab_atoms["adslab_atoms"], position_predictions):
            atoms.set_positions(positions)
        adslab_results["relaxed_atoms_" + column_name] = adslab_atoms["adslab_atoms"]
    else:
        energy_predictions = BOCPP.direct_prediction(graphs_dict["adslab_graphs"])

    adslab_results[column_name] = energy_predictions

    # Identify the best configuration and energy and save that too
    if len(energy_predictions) > 0:
        best_energy = np.min(energy_predictions)
        best_atoms = adslab_atoms["adslab_atoms"][np.argmin(energy_predictions)].copy()
        adslab_results["min_" + column_name] = best_energy
        best_atoms.set_calculator(
            SinglePointCalculator(
                atoms=best_atoms,
                energy=best_energy,
                forces=None,
                stresses=None,
                magmoms=None,
            )
        )
        adslab_results["atoms_min_" + column_name] = best_atoms
    else:
        adslab_results["min_" + column_name] = np.nan
        adslab_results["atoms_min_" + column_name] = None

    return adslab_results
