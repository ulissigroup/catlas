import yaml
from catlas.load_bulk_structures import load_ocdata_bulks
from catlas.filters import bulk_filter, adsorbate_filter, slab_filter
from catlas.load_adsorbate_structures import load_ocdata_adsorbates
from catlas.enumerate_slabs_adslabs import (
    enumerate_slabs,
    enumerate_adslabs,
    merge_surface_adsorbate_combo,
)
from catlas.dask_utils import (
    split_balance_df_partitions,
    bag_split_individual_partitions,
)

from catlas.adslab_predictions import (
    direct_energy_prediction,
    relaxation_energy_prediction,
    pop_keys,
)
import dask.bag as db
import dask
import sys
import dask.dataframe as ddf
from joblib import Memory
import pandas as pd
from dask.distributed import wait

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
    filtered_catalyst_bag = filtered_catalyst_df.to_bag(format="dict").persist()
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

    # Enumerate slab - adsorbate combos
    if config["dask"]["partitions"] == -1:
        npartitions = min(bulk_num * adsorbate_num * 4, 10000)
    else:
        npartitions = config["dask"]["partitions"]

    surface_adsorbate_combo_bag = surface_bag.product(filtered_adsorbate_bag)
    surface_adsorbate_combo_bag = surface_adsorbate_combo_bag.repartition(
        npartitions=npartitions
    )

    adslab_atoms_bag = surface_adsorbate_combo_bag.map(memory.cache(enumerate_adslabs))
    results_bag = surface_adsorbate_combo_bag.map(merge_surface_adsorbate_combo)

    # Run adslab predictions
    if "adslab_prediction_steps" in config:
        for step in config["adslab_prediction_steps"]:
            if step["step"] == "predict":
                if step["type"] == "direct" and step["gpu"] == True:
                    with dask.annotate(
                        executor="gpu", resources={"GPU": 1}, priority=10
                    ):
                        results_bag = results_bag.map(
                            memory.cache(
                                direct_energy_prediction, ignore=["batch_size"]
                            ),
                            adslab_atoms=adslab_atoms_bag,
                            config_path=step["config_path"],
                            checkpoint_path=step["checkpoint_path"],
                            column_name=step["label"],
                            batch_size=step["batch_size"],
                            cpu=False,
                        )

                elif step["type"] == "direct" and step["gpu"] == False:
                    results_bag = results_bag.map(
                        memory.cache(direct_energy_prediction, ignore=["batch_size"]),
                        adslab_atoms=adslab_atoms_bag,
                        config_path=step["config_path"],
                        checkpoint_path=step["checkpoint_path"],
                        column_name=step["label"],
                        batch_size=step["batch_size"],
                        cpu=True,
                    )

                elif step["type"] == "relaxation":
                    surface_adsorbate_combo_bag = adslab_bag.map(
                        memory.cache(relaxation_energy_prediction),
                        config_path=step["config_path"],
                        checkpoint_path=step["checkpoint_path"],
                        column_name=step["label"],
                    )
                else:
                    print("Unsupported prediction type: %s" % step["type"])

                most_recent_step = "min_" + step["label"]

    # Remove the slab and adslab atoms to make the resulting item much smaller
    # with dask.annotate(priority=10):
    #    adslab_bag = adslab_bag.map(
    #        pop_keys, keys=["adslab_graphs", "adslab_atoms", "slab_surface_object"]
    #    )

    verbose = (
        "verbose" in config["output_options"] and config["output_options"]["verbose"]
    )
    pickle = "pickle_path" in config["output_options"]

    if verbose or pickle:
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
        results = results_bag.persist(optimize_graph=False)
        wait(results)
