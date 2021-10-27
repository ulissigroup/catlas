from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import copy
from ocdata.combined import Combined
from ase.optimize import LBFGS
import numpy as np
from ocpmodels.preprocessing import AtomsToGraphs
import yaml
import copy
from ocpmodels.datasets.trajectory_lmdb import data_list_collater

from ocpmodels.common.utils import (
    radius_graph_pbc,
    setup_imports,
    setup_logging,
)

from torch.utils.data import DataLoader

BOCPP = None
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
        graph.sid = idx
        graph.fid = idx
        return graph


class BatchOCPPredictor:
    def __init__(self, config_yml, checkpoint, batch_size=8, cpu=False):

        setup_imports()
        setup_logging()

        config = yaml.safe_load(open(config_yml, "r"))
        if "includes" in config:
            for include in config["includes"]:
                include_config = yaml.safe_load(open(include, "r"))
                config.update(include_config)

        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = checkpoint
        self.batch_size = batch_size

        # tweak!
        self.config["optim"]["num_workers"] = 0
        self.config["trainer"] = "energy"

        self.trainer = registry.get_trainer_class(self.config.get("trainer", "simple"))(
            task=self.config["task"],
            model=self.config["model"],
            dataset=self.config["dataset"],
            optimizer=self.config["optim"],
            identifier="",
            slurm=self.config.get("slurm", {}),
            local_rank=self.config.get("local_rank", 0),
            is_debug=self.config.get("is_debug", True),
            cpu=cpu,
            ocp_calc=True,
        )

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

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

    def full_predict(self, graphs_list):

        graphs_list_dataset = GraphsListDataset(graphs_list)

        data_loader = self.trainer.get_dataloader(
            graphs_list_dataset,
            self.trainer.get_sampler(
                graphs_list_dataset, self.batch_size, shuffle=False
            ),
        )

        predictions = self.trainer.predict(
            data_loader, per_image=True, disable_tqdm=True
        )

        return predictions["energy"]


def direct_energy_prediction(
    adslab_dict, config_path, checkpoint_path, column_name, batch_size=8, cpu=False
):

    adslab_results = copy.copy(adslab_dict)

    global BOCPP

    if BOCPP is None:
        BOCPP = BatchOCPPredictor(
            config_yml=config_path,
            checkpoint=checkpoint_path,
            batch_size=batch_size,
            cpu=cpu,
        )

    predictions_list = BOCPP.full_predict(adslab_results["adslab_graphs"])

    adslab_results[column_name] = predictions_list

    if len(predictions_list) > 0:
        adslab_results["min_" + column_name] = min(predictions_list)
    else:
        adslab_results["min_" + column_name] = np.nan

    return adslab_results


def direct_energy_partition_prediction(
    list_of_adslab_dicts,
    config_path,
    checkpoint_path,
    column_name,
    batch_size=8,
    cpu=False,
):

    list_of_adslab_results = copy.deepcopy(list_of_adslab_dicts)

    structures = []
    partition_index = []
    for i, adslab_dict in enumerate(list_of_adslab_results):
        structures += adslab_dict["adslab_atoms"]
        partition_index += [i] * len(adslab_dict["adslab_atoms"])

    partition_index = np.array(partition_index)

    global BOCPP

    if BOCPP is None:
        BOCPP = BatchOCPPredictor(
            config_yml=config_path,
            checkpoint=checkpoint_path,
            batch_size=batch_size,
            cpu=cpu,
        )

    print(
        "Predicting %d structures across %d adslab combos!"
        % (len(structures), len(list_of_adslab_results))
    )

    predictions_list = BOCPP.full_predict(structures)

    for i, adslab_results in enumerate(list_of_adslab_results):
        relevant_predictions = predictions_list[partition_index == i]
        adslab_results[column_name] = relevant_predictions

        if len(relevant_predictions) > 0:
            adslab_results["min_" + column_name] = min(relevant_predictions)
        else:
            adslab_results["min_" + column_name] = np.nan

    return list_of_adslab_results


def relaxation_energy_prediction(
    adslabs_dict, config_path, checkpoint_path, column_name
):

    adslab_results = copy.deepcopy(adslab_dict)

    global relax_calc

    if relax_calc is None:
        relax_calc = OCPCalculator(config_path, checkpoint=checkpoint_path)

    predictions_list = []

    for adslab in adslab_results["adslab_atoms"]:
        adslab = adslab.copy()
        adslab.set_calculator(relax_calc)
        opt = LBFGS(
            adslab,
            maxstep=0.04,
            memory=1,
            damping=1.0,
            alpha=70.0,
            trajectory=None,
        )
        opt.run(fmax=0.05, steps=200)
        predictions_list.append(adslab.get_potential_energy())

    adslab_results[column_name] = predictions_list
    adslab_results["min_" + column_name] = min(predictions_list)

    return adslab_results
