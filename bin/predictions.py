import yaml
from dask_predictions.load_bulk_structures import load_ocdata_bulks
from dask_predictions.filters import bulk_filter, adsorbate_filter, slab_filter
from dask_predictions.load_adsorbate_structures import load_ocdata_adsorbates
from dask_predictions.enumerate_slabs_adslabs import enumerate_slabs, enumerate_adslabs
from dask_predictions.dask_utils import dataframe_split_individual_partitions
from dask_predictions.adslab_predictions import (
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
    exec(config["dask_config"])

    # Set up joblib memory to use for caching hard steps
    memory = Memory(config["memory_cache_location"], verbose=0)

    # Load and filter the bulks
    bulks_delayed = dask.delayed(memory.cache(load_ocdata_bulks))()
    bulk_bag = db.from_delayed([bulks_delayed])
    bulk_df = bulk_bag.to_dataframe().persist()
    print("Number of bulks: %d" % bulk_df.shape[0].compute())

    filtered_catalyst_df = bulk_filter(config, bulk_df)
    filtered_catalyst_df = dataframe_split_individual_partitions(
        filtered_catalyst_df
    ).persist()
    print("Number of filtered bulks: %d" % filtered_catalyst_df.shape[0].compute())

    # Load and filter the adsorbates
    adsorbate_delayed = dask.delayed(load_ocdata_adsorbates)()
    adsorbate_bag = db.from_delayed([adsorbate_delayed])
    adsorbate_df = adsorbate_bag.to_dataframe()
    filtered_adsorbate_df = adsorbate_filter(config, adsorbate_df)
    filtered_adsorbate_df = filtered_adsorbate_df.persist()
    print(
        "Number of filtered adsorbates: %d" % filtered_adsorbate_df.shape[0].compute()
    )

    # Enumerate surfaces
    filtered_catalyst_df["surfaces"] = filtered_catalyst_df.bulk_atoms.apply(
        memory.cache(enumerate_slabs), meta=("surfaces", "object")
    )
    filtered_catalyst_df = filtered_catalyst_df.explode("surfaces")
    filtered_catalyst_df = ddf.concat(
        [
            filtered_catalyst_df.drop(["surfaces"], axis=1),
            filtered_catalyst_df["surfaces"].apply(
                pd.Series,
                meta={
                    "slab_surface_object": "object",
                    "slab_millers": "object",
                    "slab_max_miller_index": "int",
                    "slab_shift": "float",
                    "slab_top": "bool",
                    "slab_natoms": "int",
                },
            ),
        ],
        axis=1,
    ).persist()
    print("Number of surfaces: %d" % filtered_catalyst_df.shape[0].compute())

    # Filter and repartition the surfaces
    filtered_catalyst_df = slab_filter(config, filtered_catalyst_df)
    filtered_catalyst_df = filtered_catalyst_df.persist()
    print("Number of filtered surfaces: %d" % filtered_catalyst_df.shape[0].compute())

    # Enumerate surface_adsorbate combinations
    filtered_catalyst_df = filtered_catalyst_df.assign(key=1).merge(
        filtered_adsorbate_df.assign(key=1), how="outer", on="key"
    )
    filtered_catalyst_df = dataframe_split_individual_partitions(filtered_catalyst_df)
    filtered_catalyst_df["adslabs"] = filtered_catalyst_df.apply(
        memory.cache(enumerate_adslabs), meta=("adslabs", "object"), axis=1
    )
    filtered_catalyst_df = filtered_catalyst_df.persist()
    print("Number of adslab combos: %d" % filtered_catalyst_df.shape[0].compute())

    # Run adslab predictions
    for step in config["adslab_prediction_steps"]:
        if step["step"] == "predict":
            if step["type"] == "direct":
                filtered_catalyst_df[step["label"]] = filtered_catalyst_df[
                    "adslabs"
                ].apply(
                    memory.cache(direct_energy_prediction),
                    meta=("adslab_dE", "object"),
                    config_path=step["config_path"],
                    checkpoint_path=step["checkpoint_path"],
                )
                filtered_catalyst_df["min_" + step["label"]] = filtered_catalyst_df[
                    step["label"]
                ].apply(min)
            elif step["type"] == "relaxation":
                filtered_catalyst_df[step["label"]] = filtered_catalyst_df[
                    "adslabs"
                ].apply(
                    memory.cache(relaxation_energy_prediction),
                    meta=("adslab_dE", "object"),
                    config_path=step["config_path"],
                    checkpoint_path=step["checkpoint_path"],
                )
                filtered_catalyst_df["min_" + step["label"]] = filtered_catalyst_df[
                    step["label"]
                ].apply(min)
            else:
                print("Unsupported prediction type: %s" % step["type"])

    results = filtered_catalyst_df.compute()
    print(results[['bulk_elements', 'bulk_mpid','slab_millers','adsorbate_smiles','min_dE_gemnet_is2re_finetuned']])
