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


def pop_surface_adslab_atoms(adslab_dict):
    adslab_dict = copy.deepcopy(adslab_dict)
    adslab_dict.pop("slab_surface_object")
    adslab_dict.pop("adslab_atoms")
    return adslab_dict


from torch.utils.data import Dataset
from ocpmodels.common.registry import registry

class AtomsListDataset(Dataset):
    def __init__(self, atoms_list, a2g):
        self.atoms_list = atoms_list
        self.a2g = a2g

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
        graph = self.a2g.convert(self.atoms_list[idx])
        graph.sid = idx
        graph.fid = idx
        return graph

class BatchOCPPredictor:
    def __init__(self, config_yml, checkpoint, cutoff, max_neighbors, batch_size=8, cpu=False):

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
        #self.config["optim"]["num_workers"] = 0

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

        self.a2g = AtomsToGraphs(
            max_neigh=max_neighbors,
            radius=cutoff,
            r_energy=False,
            r_forces=False,
            r_distances=False,
        )

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

    def full_predict(self, atoms_list):

        atoms_list_dataset = AtomsListDataset(atoms_list, self.a2g)

        data_loader = self.trainer.get_dataloader(
            atoms_list_dataset, self.trainer.get_sampler(atoms_list_dataset, self.batch_size, shuffle=False)
        )

        predictions = self.trainer.predict(data_loader, per_image=True, disable_tqdm=True)

        return predictions


def direct_energy_prediction(
    adslab_dict, config_path, checkpoint_path, column_name, batch_size=8, cpu=False
):

    adslab_results = copy.deepcopy(adslab_dict)

    global BOCPP

    if BOCPP is None:
        BOCPP = BatchOCPPredictor(
            config_yml=config_path,
            checkpoint=checkpoint_path,
            cutoff=6,
            max_neighbors=50,
            batch_size=batch_size,
            cpu=cpu
        )

    predictions_list = BOCPP.full_predict(adslab_results["adslab_atoms"])

    adslab_results[column_name] = predictions_list

    if len(predictions_list) > 0:
        adslab_results["min_" + column_name] = min(predictions_list)
    else:
        adslab_results["min_" + column_name] = np.nan

    return adslab_results


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
