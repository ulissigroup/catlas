import yaml
from catlas.load_bulk_structures import load_ocdata_bulks
from catlas.filters import bulk_filter, adsorbate_filter, slab_filter
from catlas.load_adsorbate_structures import load_ocdata_adsorbates
from catlas.enumerate_slabs_adslabs import enumerate_slabs, enumerate_adslabs
from catlas.dask_utils import split_balance_df_partitions

from catlas.adslab_predictions import (
    direct_energy_prediction,
    relaxation_energy_prediction,
)
import dask.bag as db
import dask
import sys
import dask.dataframe as ddf
from joblib import Memory
import pandas as pd

# Load inputs and define global vars
if __name__ == "__main__":

    # Load the config yaml
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # set up the dask cluster using the config block
    exec(config["dask"]["config"])

    # Set up joblib memory to use for caching hard steps
    memory = Memory(config["memory_cache_location"], verbose=0)

    # Load and filter the bulks
    bulks_delayed = dask.delayed(memory.cache(load_ocdata_bulks))()
    bulk_bag = db.from_delayed([bulks_delayed])
    bulk_df = bulk_bag.to_dataframe().persist()
    print("Number of bulks: %d" % bulk_df.shape[0].compute())

    filtered_catalyst_df = bulk_filter(config, bulk_df)
    bulk_num = filtered_catalyst_df.shape[0].compute()
    print("Number of filtered bulks: %d" % bulk_num)
    filtered_catalyst_bag = filtered_catalyst_df.to_bag().persist()
    bulk_num = 3
    # Load and filter the adsorbates
    adsorbate_delayed = dask.delayed(load_ocdata_adsorbates)()
    adsorbate_bag = db.from_delayed([adsorbate_delayed])
    adsorbate_df = adsorbate_bag.to_dataframe()
    filtered_adsorbate_df = adsorbate_filter(config, adsorbate_df)
    filtered_adsorbate_bag = filtered_adsorbate_df.to_bag()
    print(filtered_adsorbate_bag)
    print(
        "Number of filtered adsorbates: %d" % filtered_adsorbate_df.shape[0].compute()
    )

    # Enumerate surfaces


    surface_bag = (
        filtered_catalyst_bag.map(memory.cache(enumerate_slabs))
        .flatten()
        
    )  # WOULD BE NICE TO MAINTAIN SOME OF ZACK'S NICE PARTITIONING

    # Enumerate slab - adsorbate combos
    if config["dask"]["partitions"] == -1:
        num_partitions = min(bulk_num * 70, 10000)
    else:
        num_partitions = config["dask"]["partitions"]
    surface_adsorbate_combo_bag = surface_ads_combos = surface_bag.product(
        adsorbate_bag
    ).repartition(npartitions=num_partitions).persist()
    # Filter and repartition the surfaces ??

    adslab_bag = surface_adsorbate_combo_bag.map(memory.cache(enumerate_adslabs))

    # Run adslab predictions
    if "adslab_prediction_steps" in config:
        for step in config["adslab_prediction_steps"]:
            if step["step"] == "predict":
                if step["type"] == "direct":
                    predictions_bag = adslab_bag.map(
                        memory.cache(direct_energy_prediction),
                        config_path=step["config_path"],
                        checkpoint_path=step["checkpoint_path"],
                    )

                elif step["type"] == "relaxation":
                    predictions_bag = adslab_bag.map(
                        memory.cache(relaxation_energy_prediction),
                        config_path=step["config_path"],
                        checkpoint_path=step["checkpoint_path"],
                    )
                else:
                    print("Unsupported prediction type: %s" % step["type"])

    #                 most_recent_step = "min_" + step["label"]

    results = filtered_catalyst_df.compute()

    verbose = (
        "verbose" in config["output_options"] and config["output_options"]["verbose"]
    )
    pickle = "pickle_path" in config["output_options"]

    if verbose or pickle:
        results = predictions_bag.compute()
        df_results = pd.DataFrame(results)

        if verbose:

            print(
                df_results[
                    [
                        "composition",
                        "mpid",
                        "source" ,
                        "slab_millers",
                        "adsorbate_smile",
                        "inferred_E",
                    ]
                ]
            )

        if pickle:
            pickle_path = config["output_options"]["pickle_path"]
            if pickle_path != "None":
                results.drop(
                    [
                        "slab_surface_object",
                    ],
                    axis=1,
                ).to_pickle(pickle_path)

    else:
        results = predictions_bag.persist()
