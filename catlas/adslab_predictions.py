import copy
import os

import numpy as np
import ocpmodels
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from catlas.flag_systems import DetectTrajAnomaly
from torch.utils.data import Dataset
from tqdm import tqdm

import catlas.cache_utils
import catlas.dask_utils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation import ml_relaxation
from ocpmodels.common.utils import (
    setup_imports,
    setup_logging,
)


BOCPP_dict = {}
relax_calc = None


def pop_keys(adslab_dict, keys):
    adslab_dict = copy.deepcopy(adslab_dict)
    for key in keys:
        adslab_dict.pop(key)
    return adslab_dict


class GraphsListDataset(Dataset):
    """
    Make a list of graphs to feed into ocp dataloader object

    Extends:
        torch.utils.data.Dataset: a torch Dataset
    """

    def __init__(self, graphs_list):
        self.graphs_list = graphs_list

    def __len__(self):
        return len(self.graphs_list)

    def __getitem__(self, idx):
        graph = self.graphs_list[idx]
        return graph


class BatchOCPPredictor:
    """
    Variable used to store the model used during predictions. Specifications are
    contained in the input config. This variable is globalized within a worker to
    avoid duplicate model loading.
    """

    def __init__(self, checkpoint, number_steps, batch_size=8, cpu=False):
        self.number_steps = number_steps
        setup_imports()
        setup_logging()

        checkpoint = os.path.join(
            os.path.dirname(catlas.__file__), os.pardir, checkpoint
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
        """
        Make the dataloader used to feed graphs into the OCP model.

        Args:
            graphs_list (Iterable[torch_geometric.data.Data]): structures to run predictions on.

        Returns:
            torch.utils.data.DataLoader: an object that feeds data into pytorch models.
        """
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
            checkpoint_path (str): Path to trained model
        """
        try:
            self.trainer.load_checkpoint(checkpoint_path)
        except NotImplementedError:
            logging.warning(
                "Unable to load checkpoint!"
            )  # `logging` defined by `setup_logging()`

    def direct_prediction(self, graphs_list):
        """
        Run direct energy predictions on a list of graphs. Predict the relaxed
        energy without running ML relaxations.

        Args:
            graphs_list (Iterable[torch_geometric.data.Data]): structures to run
                inference on.

        Returns:
            Iterable[float]: predicted energies of the input structures.
        """
        data_loader = self.make_dataloader(graphs_list)

        # Batch inference
        predictions = self.trainer.predict(
            data_loader, per_image=True, disable_tqdm=True
        )

        return predictions["energy"]

    def relaxation_prediction(self, graphs_list):
        """
        Relax structures, then predict their energies.

        Args:
            graphs_list (Iterable[torch_geometric.data.Data]): structures to
                run predictions on.

        Returns:
            Iterable[float]: predicted energies
            Iterable[torch_geometric.data.Data): relaxed structures
        """
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
                steps=self.number_steps,
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
    hash_adslab_atoms,
    hash_adslab_dict,
    graphs_dict,
    checkpoint_path,
    column_name,
    batch_size=8,
    gpu_mem_per_sample=None,
    number_steps=200,
):
    """
    Predict the energies of adslabs.

    Args:
        adslab_dict (dict): A view of the adslabs to predict that will be used
            to store results. Should be ignored during cache comparisons.
        adslab_atoms (Iterable[ase.atoms.Atoms]): Adslabs containing the same
            adsorbate placed on different sites on the same surface. Should be
            ignored during cache comparisons.
        hash_adslab_atoms (str): A hash specific to `adslab_atoms` used for cache
            comparisons.
        hash_adslab_dict (str): A hash specific to `adslab_dict` used for cache
            comparisons
        graphs_dict (dict): A view of the adslabs to predict that will be used to
            predict energies. Should be ignored during cache comparisons.
        checkpoint_path (str): The path where the OCP model checkpoint can be found.
        column_name (str): An arbitrary name used to define the model output field names.
        batch_size (int, optional): The number of adslabs loaded onto a gpu at
            a single time during relaxations. Should be ignored during cache comparisons.
            Defaults to 8.
        gpu_mem_per_sample (float, optional): The approximate memory used by a single
            adslab during relaxations, used to determine batch size. Defaults to None.
        number_steps (int, optional): The number of steps used during relaxations.
            Should be determined using `bin/optimize_frame.py`. Defaults to 200.

    Returns:
        dict: Energy predictions and metadata including associated structures
            and minimum predicted energy per surface.
    """
    adslab_results = copy.copy(adslab_dict)

    global BOCPP_dict

    cpu = torch.cuda.device_count() == 0

    if (checkpoint_path, batch_size, cpu) not in BOCPP_dict:
        BOCPP_dict[checkpoint_path, batch_size, cpu] = BatchOCPPredictor(
            checkpoint=checkpoint_path,
            batch_size=batch_size,
            cpu=cpu,
            number_steps=number_steps,
        )

    BOCPP = BOCPP_dict[checkpoint_path, batch_size, cpu]
    relaxation = BOCPP.config["trainer"] == "forces"

    if "filter_reason" in adslab_dict:
        adslab_results[column_name] = []
        adslab_results["min_" + column_name] = np.nan
        adslab_results["atoms_min_" + column_name + "_initial"] = None
        if relaxation:
            adslab_results["atoms_min_" + column_name + "_relaxed"] = None

        return adslab_results
    else:
        adslab_atoms = copy.deepcopy(adslab_atoms)
        adslab_dict = copy.deepcopy(adslab_dict)

        if not cpu and gpu_mem_per_sample is not None:
            batch_size = int(
                torch.cuda.get_device_properties(0).total_memory
                / gpu_mem_per_sample
                / 1024**3
            )

        if relaxation:
            energy_predictions, position_predictions = BOCPP.relaxation_prediction(
                graphs_dict["adslab_graphs"],
            )
            energy_predictions = np.array([p.cpu().numpy() for p in energy_predictions])

            # Use the relaxed positions to generate relaxed atoms objects
            adslab_atoms_copy = copy.deepcopy(adslab_atoms)
            idx = 0
            anomaly_tests = []
            for atoms, positions in zip(adslab_atoms_copy, position_predictions):
                atoms.set_positions(positions)
                detector = DetectTrajAnomaly(
                    adslab_atoms[idx], atoms, adslab_atoms[idx].get_tags()
                )
                status = {}
                status["dissociation"] = detector.is_adsorbate_dissociated()
                status["desorption"] = detector.is_adsorbate_desorbed()
                status["reconstruction"] = detector.has_surface_changed()
                anomaly_tests.append(status)
                idx += 1
            adslab_results["relaxed_atoms_" + column_name] = adslab_atoms_copy
            adslab_results["unrelaxed_atoms_" + column_name] = adslab_atoms
            adslab_results["anomaly_detection"] = anomaly_tests
        else:
            energy_predictions = BOCPP.direct_prediction(graphs_dict["adslab_graphs"])

        adslab_results[column_name] = energy_predictions

        # Identify the best configuration and energy and save that too
        if len(energy_predictions) > 0:
            best_energy = np.min(energy_predictions)
            best_atoms_initial = adslab_atoms[np.argmin(energy_predictions)].copy()
            adslab_results["min_" + column_name] = best_energy
            best_atoms_initial.set_calculator(
                SinglePointCalculator(
                    atoms=best_atoms_initial,
                    energy=best_energy,
                    forces=None,
                    stresses=None,
                    magmoms=None,
                )
            )
            adslab_results["atoms_min_" + column_name + "_initial"] = best_atoms_initial
            # If relaxing, save the best relaxed configuration
            if relaxation:
                best_atoms_relaxed = adslab_atoms_copy[
                    np.argmin(energy_predictions)
                ].copy()
                best_atoms_relaxed.set_calculator(
                    SinglePointCalculator(
                        atoms=best_atoms_relaxed,
                        energy=best_energy,
                        forces=None,
                        stresses=None,
                        magmoms=None,
                    )
                )
                adslab_results[
                    "atoms_min_" + column_name + "_relaxed"
                ] = best_atoms_relaxed
        else:
            adslab_results[column_name] = []
            adslab_results["min_" + column_name] = np.nan
            adslab_results["atoms_min_" + column_name + "_initial"] = None
            if relaxation:
                adslab_results["atoms_min_" + column_name + "_relaxed"] = None

        return adslab_results


def count_steps(config, df_results):
    """Count the number of adslabs remaining at each filtering step.

    Args:
        config (dict): a catlas config
        df_results (pd.core.frame.DataFrame): the output of a catlas run

    Returns:
        list[dict]: a list of steps and the number of adslabs remaining at that step
        int: the original number of adslabs before filtering
    """
    inference_list = []
    for step in config["adslab_prediction_steps"]:
        if step["step_type"] == "inference":
            counts = sum(
                df_results[~df_results["min_" + step["label"]].isnull()][
                    step["label"]
                ].apply(len)
            )
            label = step["label"]
            inference_list.append({"counts": counts, "label": label})
    num_adslabs: int = sum(df_results[inference_list[0]["label"]].apply(len))
    return inference_list, num_adslabs