import os
import sys
import time
import warnings
import dask
import dask.sizeof
import joblib
import pandas as pd
import yaml
from dask import bag as db
from dask.distributed import wait
from jinja2 import Template
import argparse

import catlas.dask_utils
from catlas.adslab_predictions import energy_prediction
from catlas.config_validation import config_validator
from catlas.dask_utils import bag_split_individual_partitions, to_pickles
from catlas.enumerate_slabs_adslabs import (
    convert_adslabs_to_graphs,
    enumerate_adslabs,
    enumerate_slabs,
    merge_surface_adsorbate_combo,
)
from catlas.filter_utils import pb_query_and_write, get_first_type
from catlas.filters import (
    adsorbate_filter,
    bulk_filter,
    predictions_filter,
    slab_filter,
)
from catlas.load_adsorbate_structures import load_ocdata_adsorbates
from catlas.load_bulk_structures import load_bulks
from catlas.nuclearity import get_nuclearity
from catlas.parity.parity_utils import get_parity_upfront
from catlas.sankey.sankey_utils import Sankey


def parse_inputs():
    """bin/predictions.py takes in two command-line inputs: a path to a config file
    describing how predictions are run, and a path to a dask cluster script that
    determines how to connect to a dask cluster that will execute computations. This
    script validates the config file and runs the dask cluster script, returning the
    config and dask cluster script for further use by the main script.

    Returns:
        dict: a config describing what adsorption predictions to run.
    """
    parser = argparse.ArgumentParser(description="Predict adsorption energies.")
    parser.add_argument(
        "config_path",
        type=str,
        help="""A path to a config yml file describing what adsorption predictions to
        run.""",
    )
    parser.add_argument(
        "dask_cluster_script_path",
        type=str,
        help="""A path to a script defining how to connect to a dask cluster that will
        run calculations.""",
    )

    args = parser.parse_args()
    config_path, dask_cluster_script_path = (
        args.config_path,
        args.dask_cluster_script_path,
    )

    template = Template(open(config_path).read())
    config = yaml.load(template.render(**os.environ), Loader=yaml.FullLoader)
    if config.get("validate", True) and not config_validator.validate(config):
        raise ValueError(
            f"""Config has the following errors:{os.linesep}{
                os.linesep.join(
                    [
                        ": ".join([f'"{i}"' for i in item])
                        for item in config_validator.errors.items()
                    ]
            )
            }"""
        )
    else:
        print("Config validated")

    # Start the dask cluster
    dask_cluster_script = Template(open(dask_cluster_script_path).read()).render(
        **os.environ
    )

    return config, dask_cluster_script


def load_bulks_and_filter():
    pass


def load_adsorbates_and_filter():
    pass


def enumerate_surfaces_and_filter():
    pass


def enumerate_and_predict_adslabs():
    pass


def generate_outputs():
    pass


# Load inputs and define global vars
if __name__ == "__main__":
    """Run predictions according to input config file

    Usage (see examples in `.github/workflows/automated_screens`):
        If cluster is not running, start it:
            kubectl apply -f \
                configs/dask_cluster/dask_operator/catlas-hybrid-cluster.yml
            kubectl scale --replicas=4 daskworkergroup \
                catlas-hybrid-cluster-default-worker-group
            kubectl scale --replicas=1 daskworkergroup \
                catlas-hybrid-cluster-gpu-worker-group
        python bin/predictions.py path/to/config.yml \
            configs/dask_cluster/dask_operator/dask_connect.py

    Args:
        config (str): File path where a config is found. Example configs can be
            found in `configs/automated_screens` and
            `.github/workflows/automated_screens`
        dask_connect_script (str): script to connect to running dask cluster
    Raises:
        ValueError: The provided config is invalid.
    """
    # Load the config yaml
    config, dask_cluster_script = parse_inputs()

    # Establish run information
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + config["output_options"]["run_name"]
    os.makedirs(f"outputs/{run_id}/")

    # Print catlas to terminal
    with open("catlas/catlas_ascii.txt", "r") as f:
        print(f.read())

    # Generate parity plots
    if ("make_parity_plots" in config["output_options"]) and (
        config["output_options"]["make_parity_plots"]
    ):
        get_parity_upfront(config, run_id)
        print(
            """Parity plots are ready if data was available, please review them to
                ensure the model selected meets your needs."""
        )

    exec(dask_cluster_script)

    # Load the bulks
    bulks_delayed = dask.delayed(
        catlas.cache_utils.sqlitedict_memoize(
            config["memory_cache_location"], load_bulks
        )
    )(config["input_options"]["bulk_file"])
    bulk_bag = db.from_delayed([bulks_delayed])
    bulk_df = bulk_bag.to_dataframe().repartition(npartitions=100).persist()

    # Create pourbaix lmdb if it doesnt exist
    if "filter_by_pourbaix_stability" in list(config["bulk_filters"].keys()):
        lmdb_path = config["bulk_filters"]["filter_by_pourbaix_stability"]["lmdb_path"]
        if not os.path.isfile(lmdb_path):
            warnings.warn(
                "No lmdb was found here:" + lmdb_path + ". Making the lmdb instead."
            )
            bulk_bag = bulk_bag.repartition(npartitions=200)
            pbx_dicts = bulk_bag.map(pb_query_and_write, lmdb_path=lmdb_path).compute()

    # Instantiate Sankey dictionary
    sankey = Sankey(
        {
            "label": ["Adsorbates from db", "Bulks from db"],
            "source": [],
            "target": [],
            "value": [],
            "x": [0.001, 0.001],
            "y": [0.001, 0.8],
        }
    )

    # Filter the bulks
    initial_bulks = bulk_df.shape[0].compute()
    print(f"Number of initial bulks: {initial_bulks}")

    # Force dask to distribute the bulk objects across all workers
    # Two rebalances were necessary for some reason.
    wait(bulk_df)
    client.rebalance(bulk_df)  # `client` is defined during `exec(dask_cluster_script)`
    client.rebalance(bulk_df)

    # Filter the bulks
    filtered_catalyst_df, sankey = bulk_filter(config, bulk_df, sankey, initial_bulks)
    bulk_num = filtered_catalyst_df.shape[0].compute()
    print("Number of filtered bulks: %d" % bulk_num)
    filtered_catalyst_bag = filtered_catalyst_df.to_bag(format="dict").persist()

    # partition to 1 bulk per partition
    filtered_catalyst_bag = bag_split_individual_partitions(filtered_catalyst_bag)

    # Load and filter the adsorbates
    adsorbate_delayed = dask.delayed(load_ocdata_adsorbates)(
        config["input_options"]["adsorbate_file"]
    )
    adsorbate_df = db.from_delayed([adsorbate_delayed]).to_dataframe()
    filtered_adsorbate_df, sankey = adsorbate_filter(config, adsorbate_df, sankey)
    adsorbate_num = filtered_adsorbate_df.shape[0].compute()
    filtered_adsorbate_bag = filtered_adsorbate_df.to_bag(format="dict")
    print("Number of filtered adsorbates: %d" % adsorbate_num)

    # Enumerate and filter surfaces
    unfiltered_surface_bag = (
        filtered_catalyst_bag.map(
            catlas.cache_utils.sqlitedict_memoize(
                config["memory_cache_location"], enumerate_slabs
            )
        )
        .flatten()
        .persist()
    )
    surface_bag = unfiltered_surface_bag.filter(lambda x: slab_filter(config, x))
    surface_bag = surface_bag.map(get_nuclearity)

    # choose the number of partitions after to use after making adslab combos
    npartitions = min(bulk_num * 10, 1000)
    surface_bag = surface_bag.repartition(npartitions=npartitions)

    # Enumerate slab - adsorbate combos
    surface_adsorbate_combo_bag = surface_bag.product(filtered_adsorbate_bag)

    # Enumerate the adslab configs and the graphs on any worker
    adslab_atoms_bag = surface_adsorbate_combo_bag.map(
        catlas.cache_utils.sqlitedict_memoize(
            config["memory_cache_location"], enumerate_adslabs, shard_digits=4
        )
    )
    graphs_bag = adslab_atoms_bag.map(convert_adslabs_to_graphs)
    results_bag = surface_adsorbate_combo_bag.map(merge_surface_adsorbate_combo)
    hash_adslab_atoms_bag = adslab_atoms_bag.map(joblib.hash)

    # Run adslab predictions
    inference = False
    if "adslab_prediction_steps" in config:
        for step in config["adslab_prediction_steps"]:
            if "filter" in step["step_type"]:
                results_bag = results_bag.map_partitions(
                    predictions_filter,
                    step,
                )
            elif step["step_type"] == "inference" and step["gpu"]:
                inference = True
                number_steps = step["number_steps"] if "number_steps" in step else 200
                hash_results_bag = results_bag.map(joblib.hash)

                with dask.annotate(resources={"GPU": 1}, priority=10000000):
                    results_bag = results_bag.map(
                        catlas.cache_utils.sqlitedict_memoize(
                            config["memory_cache_location"],
                            energy_prediction,
                            ignore=[
                                "batch_size",
                                "gpu_mem_per_sample",
                                "graphs_dict",
                                "adslab_atoms",
                                "adslab_dict",
                            ],
                            shard_digits=4,
                        ),
                        adslab_atoms=adslab_atoms_bag,
                        hash_adslab_atoms=hash_adslab_atoms_bag,
                        hash_adslab_dict=hash_results_bag,
                        graphs_dict=graphs_bag,
                        checkpoint_path=step["checkpoint_path"],
                        column_name=step["label"],
                        batch_size=step["batch_size"],
                        gpu_mem_per_sample=step.get("gpu_mem_per_sample", None),
                        number_steps=number_steps,
                    )
            elif step["step_type"] == "inference" and not step["gpu"]:
                inference = True
                number_steps = step["number_steps"] if "number_steps" in step else 200
                hash_results_bag = results_bag.map(joblib.hash)

                results_bag = results_bag.map(
                    catlas.cache_utils.sqlitedict_memoize(
                        config["memory_cache_location"],
                        energy_prediction,
                        ignore=[
                            "batch_size",
                            "graphs_dict",
                            "adslab_atoms",
                            "adslab_dict",
                        ],
                        shard_digits=4,
                    ),
                    adslab_atoms=adslab_atoms_bag,
                    hash_adslab_atoms=hash_adslab_atoms_bag,
                    hash_adslab_dict=hash_results_bag,
                    graphs_dict=graphs_bag,
                    checkpoint_path=step["checkpoint_path"],
                    column_name=step["label"],
                    batch_size=step["batch_size"],
                    number_steps=number_steps,
                )

                most_recent_step = "min_" + step["label"]

    # Handle results
    verbose = (
        "verbose" in config["output_options"] and config["output_options"]["verbose"]
    )

    results_bag = results_bag.persist(optimize_graph=False)

    if config["output_options"]["pickle_intermediate_outputs"]:
        os.makedirs(f"outputs/{run_id}/intermediate_pkls/")
        to_pickles(
            results_bag,
            f"outputs/{run_id}/intermediate_pkls/" + "/*.pkl",
            optimize_graph=False,
        )

    if verbose or config["output_options"]["pickle_final_output"]:
        results = results_bag.compute(optimize_graph=False)
        df_results = pd.DataFrame(results)
        if inference:
            num_inferred = []
            for step in config["adslab_prediction_steps"]:
                if step["step_type"] == "inference":
                    counts = sum(
                        df_results[~df_results["min_" + step["label"]].isnull()][
                            step["label"]
                        ].apply(len)
                    )
                    label = step["label"]
                    num_inferred.append({"counts": counts, "label": label})
            num_adslabs = sum(df_results[num_inferred[0]["label"]].apply(len))
        filtered_slabs = len(df_results)
        if verbose and inference:

            print(
                df_results[
                    [
                        "bulk_elements",
                        "bulk_id",
                        "bulk_data_source",
                        "slab_millers",
                        "adsorbate_smiles",
                        most_recent_step,
                    ]
                ]
            )
        elif verbose:
            print(
                df_results[
                    [
                        "bulk_elements",
                        "bulk_id",
                        "bulk_data_source",
                        "slab_millers",
                        "adsorbate_smiles",
                    ]
                ]
            )

    else:
        # Important to use optimize_grap=False so that the information
        # on only running GPU inference on GPUs is saved
        results = results_bag.persist(optimize_graph=False)
        wait(results)
        filtered_slabs = results.count().compute()

    if config["output_options"]["pickle_final_output"]:
        pickle_path = f"outputs/{run_id}/results_df.pkl"

        if (
            "output_all_structures" in config["output_options"]
            and config["output_options"]["output_all_structures"]
        ):
            adslab_atoms = adslab_atoms_bag.compute(optimize_graph=False)
            df_results["adslab_atoms"] = adslab_atoms
            df_results.to_pickle(pickle_path)
            if not inference:
                num_adslabs = sum(df_results["adslab_atoms"].apply(len))
                filtered_slabs = len(df_results)
                num_inferred = [0]
        else:
            # screen classes from custom packages
            class_mask = (
                df_results.columns.to_series()
                .apply(
                    lambda x: str(
                        get_first_type(
                            df_results[x].iloc[df_results[x].first_valid_index()]
                        )
                    )
                    if type(df_results[x].first_valid_index()) == int
                    else str(type(df_results[x].iloc[0]))
                )
                .apply(lambda x: "catkit" in x or "ocp" in x or "ocdata" in x)
            )
            df_results[class_mask[~class_mask].index].to_pickle(pickle_path)

    with open(f"outputs/{run_id}/inputs_config.yml", "w") as fhandle:
        yaml.dump(config, fhandle)

    # Make final updates to the sankey diagram and plot it
    unfiltered_slabs = unfiltered_surface_bag.count().compute()

    if "num_adslabs" not in vars():
        num_adslabs = num_inferred = [0]
        warnings.warn(
            "Adslabs were enumerated but will not be counted for the Sankey diagram."
        )
    sankey.add_slab_info(unfiltered_slabs, filtered_slabs)

    sankey.add_adslab_info(num_adslabs, num_inferred)
    sankey.get_sankey_diagram(run_id)
