import yaml
from dask_predictions.load_bulk_structures import load_ocdata_bulks
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

    print('Total number of filtered bulks is %d' %bulk_dataframe.size)


