import os
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
from catlas.adslab_predictions import energy_prediction, count_steps
from catlas.config_validation import config_validator
from catlas.dask_utils import to_pickles
from catlas.enumerate_slabs_adslabs import (
    convert_adslabs_to_graphs,
    enumerate_adslabs,
    enumerate_slabs,
    merge_surface_adsorbate_combo,
)
from catlas.filter_utils import pb_query_and_write, filter_columns_by_type
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
    """
    Parse and prepare inputs for use by the main script. This function loads and validates
    the config, pulls the run_id from the config,generates the dask cluster script from the
    dask cluster script path, makes a folder for the ouputs, generates parity plots if
    available for the model(s) in use, generates a Sankey dictionary for later use in the
    script, and writes "CATLAS" in large isometrically displayed ASCII letters.

    Args:
        config_path (str): a path to a config yml file describing what adsorption
            calculations to run in the main script.
        dask_cluster_script_path (str): a path to a script that connects to a dask
            cluster that executes calculations run during the main script.



    Returns:
        dict: a config describing what adsorption predictions to run.
        str: the text of a python script that connects to a Dask cluster
        str: a string including a timestamp that uniquely identifies this run.
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

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + config["output_options"]["run_name"]
    os.makedirs(f"outputs/{run_id}/")

    if ("make_parity_plots" in config["output_options"]) and (
        config["output_options"]["make_parity_plots"]
    ):
        get_parity_upfront(config, run_id)
        print(
            """Parity plots are ready if data was available, please review them to
                ensure the model selected meets your needs."""
        )

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

    with open("catlas/catlas_ascii.txt", "r") as f:
        print(f.read())

    return config, dask_cluster_script, run_id, sankey


def load_bulks_and_filter(config, client, sankey):
    """
    Load bulk materials from file and filter them according to the input config
    file. Update sankey diagram based on bulk filtering.

    Args:
        config (dict): a dictionary specifying what adsorption calculations to run.
        client (dask.distributed.Client): a Dask cluster that runs calculations during
            execution of this program.
        sankey (catlas.sankey.sankey_utils.Sankey): a Sankey object describing how
            objects have been filtered so far.

    Returns:
        dict: a dictionary containing bulk materials that survived filtering
        catlas.sankey.sankey_utils.Sankey: a Sankey object updated with bulk filtering.
    """
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
            bulk_bag.map(pb_query_and_write, lmdb_path=lmdb_path).compute()

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

    filtered_catalyst_bag = filtered_catalyst_df.to_bag(format="dict").persist()
    bulk_num = filtered_catalyst_bag.count().compute()
    print("Number of filtered bulks: %d" % bulk_num)

    return filtered_catalyst_bag, sankey, bulk_num


def load_adsorbates_and_filter(config, sankey):
    """
    Load adsorbates and filter them according to the input config. Update sankey
    diagram based on adsorbate filtering.

    Args:
        config (dict): a config file specifying what adsorption calculations to run.
        sankey (catlas.sankey.sankey_utils.Sankey): a Sankey object describing how
            objects have been filtered so far.

    Returns:
        dict: a dictionary containing adsorbates that survived filtering.
        catlas.sankey.sankey_utils.Sankey: a Sankey object updated with bulk filtering.
        int: the number of bulk materials that survived filtering.
    """
    adsorbate_delayed = dask.delayed(load_ocdata_adsorbates)(
        config["input_options"]["adsorbate_file"]
    )

    adsorbate_df = db.from_delayed([adsorbate_delayed]).to_dataframe()
    filtered_adsorbate_df, sankey = adsorbate_filter(config, adsorbate_df, sankey)

    adsorbate_bag = filtered_adsorbate_df.to_bag(format="dict")
    adsorbate_num = adsorbate_bag.count().compute()
    print("Number of filtered adsorbates: %d" % adsorbate_num)

    return adsorbate_bag, sankey


def enumerate_surfaces_and_filter(config, filtered_catalyst_bag, bulk_num):
    """
    Enumerate surfaces from an input bulk bag according to the input config.

    Args:
        config (dict): A config file specifying what surfaces to filter.
        filtered_catalyst_bag (dask.bag.Bag): Bulk materials to enumerate surfaces for.
        bulk_num (int): The number of bulk materials in the input bag.

    Returns:
        dask.bag.Bag: A dask Bag containing filtered surfaces
        int: The number of slabs enumerated from filtered bulks before slab filtering.
    """
    # Enumerate and filter surfaces
    max_miller = (
        config["slab_filters"]["filter_by_max_miller_index"]
        if "filter_by_max_miller_index" in config["slab_filters"]
        else 2
    )
    unfiltered_surface_bag = (
        filtered_catalyst_bag.map(
            catlas.cache_utils.sqlitedict_memoize(
                config["memory_cache_location"], enumerate_slabs
            ),
            max_miller=max_miller,
        )
        .flatten()
        .persist()
    )
    surface_bag = unfiltered_surface_bag.map_partitions(slab_filter, config)
    surface_bag = surface_bag.map(get_nuclearity)

    npartitions = min(bulk_num * 10, 1000)

    surface_bag = surface_bag.repartition(npartitions=npartitions)
    num_unfiltered_slabs = unfiltered_surface_bag.count().compute()
    return surface_bag, num_unfiltered_slabs


def enumerate_adslabs_wrapper(
    config,
    surface_bag,
    adsorbate_bag,
):
    """
    Generate adslabs from all filtered surfaces and adsorbates.

    Args:
        config (dict): A config file specifying what surfaces to filter.
        surface_bag (dask.bag.Bag): surfaces to enumerate
        adsorbate_bag (dask.bag.Bag): adsorbates to enumerate

    Returns:
        dask.bag.Bag: a bag of metadata for the adslabs
        dask.bag.Bag: a bag of enumerated adslabs

    """

    surface_adsorbate_combo_bag = surface_bag.product(adsorbate_bag)

    adslab_atoms_bag = surface_adsorbate_combo_bag.map(
        catlas.cache_utils.sqlitedict_memoize(
            config["memory_cache_location"], enumerate_adslabs, shard_digits=4
        )
    )
    results_bag = surface_adsorbate_combo_bag.map(merge_surface_adsorbate_combo)
    return adslab_atoms_bag, results_bag


def make_predictions(
    config,
    adslab_atoms_bag,
    results_bag,
):
    """
    Make predictions on enumerated adslabs. This may include a single inference step
    or multiple with optional intermediate filtering.

    Args:
        config (dict): A config file specifying what surfaces to filter.
        results_bag (dask.bag.Bag): a bag of metadata for the adslabs
        adslab_atoms_bag (dask.bag.Bag): a bag of enumerated adslabs

    Returns:
        dask.bag.Bag: a dask Bag containing adslabs and their predicted adsorption
            energies according to models specified in the config file.
        dask.bag.Bag: a dask Bag containing adslabs before any predictions were run.
        bool: True if a model was used to predict adsorption energies of the inputs.
        str: the name of the column corresponding to the minimum adsorption energy
            on each surface according to the model that was run last during predictions.
    """
    graphs_bag = adslab_atoms_bag.map(convert_adslabs_to_graphs)
    hash_adslab_atoms_bag = adslab_atoms_bag.map(joblib.hash)
    inference = False

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
            most_recent_step = "min_" + step["label"]
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
    return adslab_atoms_bag, results_bag, inference, most_recent_step


def generate_outputs(
    config,
    adslab_atoms_bag,
    results_bag,
    run_id,
    inference,
    most_recent_step,
):
    """
    Process the remaining outputs selecting in the inputs yaml for catlas.

    Args:
        config (dict): A config file specifying what surfaces to filter.
        results_bag (dask.bag.Bag): A dask Bag object of adslabs and their predicted
            adsorption energies.
        run_id (str): A string with a timestamp uniquely identifying the run.
        inference (bool): Whether a model was used to predict adsorption energies
            during the execution of this script.

    Returns:
        int: the number of adslabs immediately after adslab enumeration
        int: the number of adslabs that inference was run on
        int: the number of adslabs remaining after all inference
    """
    output_options = config["output_options"]
    verbose = output_options["verbose"]
    compute = verbose or output_options["pickle_final_output"]
    num_adslabs = None

    if not inference:
        inference_list = [{"label": "no inference", "counts": 0}]
        num_adslabs = results_bag.map(len).sum().compute()
        num_filtered_slabs = results_bag.count().compute()

    if output_options["pickle_intermediate_outputs"]:
        os.makedirs(f"outputs/{run_id}/intermediate_pkls/")
        to_pickles(
            results_bag,
            f"outputs/{run_id}/intermediate_pkls/" + "/*.pkl",
            optimize_graph=False,
        )

    if compute:
        results = results_bag.compute(optimize_graph=False)
        df_results = pd.DataFrame(results)
        if inference:
            inference_list, num_adslabs = count_steps(config, df_results)
        num_filtered_slabs = len(df_results)
        if verbose:
            print(
                df_results[
                    [
                        "bulk_elements",
                        "bulk_id",
                        "bulk_data_source",
                        "slab_millers",
                        "adsorbate_smiles",
                        *[c for c in df_results.columns if c == most_recent_step],
                    ]
                ]
            )

    else:
        # Important to use optimize_graph=False so that the information
        # on only running GPU inference on GPUs is saved
        results = results_bag.persist(optimize_graph=False)
        wait(results)

    if output_options["pickle_final_output"]:
        pickle_path = f"outputs/{run_id}/results_df.pkl"

        if (
            "output_all_structures" in output_options
            and output_options["output_all_structures"]
        ):
            adslab_atoms = adslab_atoms_bag.compute(optimize_graph=False)
            df_results["adslab_atoms"] = adslab_atoms
            df_results.to_pickle(pickle_path)

        else:
            class_mask = filter_columns_by_type(
                df_results, type_kws=["catkit", "ocp", "ocdata"]
            )
            df_results[class_mask[~class_mask].index].to_pickle(pickle_path)

    with open(f"outputs/{run_id}/inputs_config.yml", "w") as fhandle:
        yaml.dump(config, fhandle)

    return num_adslabs, inference_list, num_filtered_slabs


def finish_sankey_diagram(
    sankey,
    num_unfiltered_slabs,
    num_filtered_slabs,
    num_adslabs,
    inference_list,
    run_id,
) -> Sankey:
    """
    Make sankey diagram for the catlas run.

    Args:
        sankey (catlas.sankey.sankey_utils.Sankey): Sankey object from predictions run
        num_unfiltered_slabs (int): the number of slabs before filtering
        num_filtered_slabs (int): the number of slabs after filtering
        num_adslabs (int): the number of adslabs enumerated
        num_inferred (int): the number of adslabs that inference was run on
        run_id (str): an arbitrary string identifying the run

    Returns:
        catlas.sankey.sankey_utils.Sankey: finished Sankey diagram
    """
    if num_adslabs is None:
        num_adslabs = 0
        inference_list = [{"label": "no inference", "counts": 0}]
        warnings.warn(
            "Adslabs were enumerated but will not be counted for the Sankey diagram."
        )
    sankey.add_slab_info(num_unfiltered_slabs, num_filtered_slabs)
    sankey.add_adslab_info(num_adslabs, inference_list)
    sankey.get_sankey_diagram(run_id)
    return sankey
