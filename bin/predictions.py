import yaml
from dask_predictions.load_bulk_structures import load_ocdata_bulks
from dask_predictions.filters import bulk_filter, adsorbate_filter
from dask_predictions.load_adsorbate_structures import load_ocdata_adsorbates
import dask.bag as db
import dask
import sys
from joblib import Memory

# Load inputs and define global vars
if __name__ == "__main__":

    # Load the config yaml
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # set up the dask cluster using the config block
    exec(config["dask_config"])

    # Set up joblib memory to use for caching hard steps
    memory = Memory(config["memory_cache_location"], verbose=1)

    # Load and filter the bulks
    bulks_delayed = dask.delayed(load_ocdata_bulks)()
    bulk_bag = db.from_delayed([bulks_delayed])
    bulk_dataframe = bulk_bag.to_dataframe()
    filtered_bulk_dataframe = bulk_filter(config, bulk_dataframe)
    filtered_bulk_dataframe = filtered_bulk_dataframe.persist()
    print(
        "Total number of filtered bulks is %d"
        % filtered_bulk_dataframe.compute().shape[0]
    )

    # Load and filter the adsorbates
    adsorbate_delayed = dask.delayed(load_ocdata_adsorbates)()
    adsorbate_bag = db.from_delayed([adsorbate_delayed])
    adsorbate_dataframe = adsorbate_bag.to_dataframe()
    adsorbate_dataframe = adsorbate_dataframe.persist()
    filtered_adsorbate_dataframe = adsorbate_filter(config, adsorbate_dataframe)
    print(
        "Total number of filtered adsorbates is %d"
        % filtered_adsorbate_dataframe.shape[0].compute()
    )
