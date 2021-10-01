import yaml
from dask_predictions.load_bulk_structures import load_ocdata_bulks
from dask_predictions.load_adsorbate_structures import load_ocdata_adsorbates
import dask.bag as db
import dask
import sys

# Load inputs and define global vars
if __name__ == "__main__":
    config_path = sys.argv[1]

    with open(config_path) as f:
        config = yaml.safe_load(f)

    bulks_delayed = dask.delayed(load_ocdata_bulks)()
    bulk_bag = db.from_delayed([bulks_delayed])
    bulk_dataframe = bulk_bag.to_dataframe()
    bulk_dataframe = bulk_dataframe.compute()

    print('Total number of bulks is %d' %bulk_dataframe.shape[0])

    adsorbate_delayed = dask.delayed(load_ocdata_adsorbates)()
    adsorbate_bag = db.from_delayed([adsorbate_delayed])
    adsorbate_dataframe = adsorbate_bag.to_dataframe()
    adsorbate_dataframe = adsorbate_dataframe.compute()

    print('Total number of adsorbates is %d' %adsorbate_dataframe.shape[0])
