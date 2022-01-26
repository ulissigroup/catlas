import yaml
from parity.parity_utils import get_parity_upfront
from catlas.load_bulk_structures import load_bulks
from catlas.filters import bulk_filter, adsorbate_filter, slab_filter
from catlas.load_adsorbate_structures import load_ocdata_adsorbates
from catlas.enumerate_slabs_adslabs import (
    enumerate_slabs,
    enumerate_adslabs,
    convert_adslabs_to_graphs,
    merge_surface_adsorbate_combo,
)
from catlas.dask_utils import (
    bag_split_individual_partitions,
    to_pickles,
)

from catlas.cache_utils import (
    better_build_func_identifier,
)
import warnings
from catlas.adslab_predictions import (
    energy_prediction,
)
import dask.bag as db
import dask
import sys
import dask.dataframe as ddf
from joblib import Memory
import pandas as pd
from dask.distributed import wait
from jinja2 import Template
import os
import pickle
import tqdm
import datetime
import joblib

joblib.memory._build_func_identifier = better_build_func_identifier


# Load inputs and define global vars
if __name__ == "__main__":

    # Load the config yaml
    config_path = sys.argv[1]
    template = Template(open(config_path).read())
    config = yaml.load(template.render(**os.environ))

    # Generate parity plots
    if "parity_output_folder" in config["output_options"]:
        get_parity_upfront(config)
        print(
            "Parity plots are ready if data was available, please review them to ensure the model selected meets your needs."
        )

    # Start the dask cluster
    dask_cluster_script = Template(open(sys.argv[2]).read()).render(**os.environ)
    exec(dask_cluster_script)

    # Set up joblib memory to use for caching hard steps
    memory = Memory(config["memory_cache_location"], verbose=0)

    # Load and filter the bulks
    bulks_delayed = dask.delayed(memory.cache(load_bulks))(
        config["input_options"]["bulk_file"]
    )
    bulk_bag = db.from_delayed([bulks_delayed])
    bulk_df = bulk_bag.to_dataframe().persist()
    print("Number of initial bulks: %d" % bulk_df.shape[0].compute())

    filtered_catalyst_df = bulk_filter(config, bulk_df)
    bulk_num = filtered_catalyst_df.shape[0].compute()
    print("Number of filtered bulks: %d" % bulk_num)
    filtered_catalyst_bag = filtered_catalyst_df.to_bag(format="dict").persist()

    # partition to 1 bulk per partition
    filtered_catalyst_bag = bag_split_individual_partitions(filtered_catalyst_bag)

    # Load and filter the adsorbates
    adsorbate_delayed = dask.delayed(load_ocdata_adsorbates)()
    adsorbate_bag = db.from_delayed([adsorbate_delayed])
    adsorbate_df = adsorbate_bag.to_dataframe()
    filtered_adsorbate_df = adsorbate_filter(config, adsorbate_df)
    adsorbate_num = filtered_adsorbate_df.shape[0].compute()
    filtered_adsorbate_bag = filtered_adsorbate_df.to_bag(format="dict")
    print("Number of filtered adsorbates: %d" % adsorbate_num)

    # Enumerate and filter surfaces
    surface_bag = filtered_catalyst_bag.map(memory.cache(enumerate_slabs)).flatten()
    surface_bag = surface_bag.filter(lambda x: slab_filter(config, x))

    # choose the number of partitions after to use after making adslab combos
    if config["dask"]["partitions"] == -1:
        npartitions = min(bulk_num * adsorbate_num * 4, 2000)
    else:
        npartitions = config["dask"]["partitions"]

    # Enumerate slab - adsorbate combos
    surface_adsorbate_combo_bag = surface_bag.product(filtered_adsorbate_bag)
    surface_adsorbate_combo_bag = surface_adsorbate_combo_bag.repartition(
        npartitions=npartitions
    )

    # Enumerate the adslab configs and the graphs on any worker
    adslab_atoms_bag = surface_adsorbate_combo_bag.map(memory.cache(enumerate_adslabs))
    graphs_bag = adslab_atoms_bag.map(convert_adslabs_to_graphs)
    results_bag = surface_adsorbate_combo_bag.map(merge_surface_adsorbate_combo)

    # Run adslab predictions
    if "adslab_prediction_steps" in config:
        for step in config["adslab_prediction_steps"]:
            if step["step"] == "predict":

                if step["gpu"]:
                    with dask.annotate(resources={"GPU": 1}, priority=10):
                        results_bag = results_bag.map(
                            memory.cache(
                                energy_prediction,
                                ignore=["batch_size", "graphs_dict", "cpu"],
                            ),
                            adslab_atoms=adslab_atoms_bag,
                            graphs_dict=graphs_bag,
                            checkpoint_path=step["checkpoint_path"],
                            column_name=step["label"],
                            batch_size=step["batch_size"],
                            cpu=False,
                        )
                else:
                    results_bag = results_bag.map(
                        memory.cache(
                            energy_prediction,
                            ignore=["batch_size", "graphs_dict", "cpu"],
                        ),
                        adslab_atoms=adslab_atoms_bag,
                        graphs_dict=graphs_bag,
                        checkpoint_path=step["checkpoint_path"],
                        column_name=step["label"],
                        batch_size=step["batch_size"],
                        cpu=True,
                    )

                most_recent_step = "min_" + step["label"]

    verbose = (
        "verbose" in config["output_options"] and config["output_options"]["verbose"]
    )
    pickle_in_config = "pickle_path" in config["output_options"]

    results_bag = results_bag.persist(optimize_graph=False)

    if "pickle_folder" in config["output_options"]:
        to_pickles(
            results_bag,
            config["output_options"]["pickle_folder"] + "/*.pkl",
            optimize_graph=False,
        )

    if verbose or pickle_in_config:
        results = results_bag.compute(optimize_graph=False)
        df_results = pd.DataFrame(results)

        if verbose:

            print(
                df_results[
                    [
                        "bulk_elements",
                        "bulk_mpid",
                        "bulk_data_source",
                        "slab_millers",
                        "adsorbate_smiles",
                        most_recent_step,
                    ]
                ]
            )

    else:
        # Important to use optimize_grap=False so that the information
        # on only running GPU inference on GPUs is saved
        results = results_bag.persist(optimize_graph=False)
        wait(results)

    if pickle_in_config:
        pickle_path = config["output_options"]["pickle_path"]

        if pickle_path != "None":
            if (
                "output_all_structures" in config["output_options"]
                and config["output_options"]["output_all_structures"]
            ):
                adslab_atoms = adslab_atoms_bag.compute(optimize_graph=False)
                df_results["adslab_atoms"] = adslab_atoms
                df_results.to_pickle(pickle_path)
            else:
                # screen classes from custom packages
                class_mask = (
                    df_results.columns.to_series()
                    .apply(lambda x: str(type(df_results[x].iloc[0])))
                    .apply(lambda x: "catkit" in x or "ocp" in x or "ocdata" in x)
                )
                df_results[class_mask[~class_mask].index].to_pickle(pickle_path)

        with open(config["output_options"]["config_path"], "w") as fhandle:
            yaml.dump(config, fhandle)
