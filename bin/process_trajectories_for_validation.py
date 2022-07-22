"""Process trajectories to summary validation pickle file."""
from catlas.parity.data_processing_utils import ProcessValTraj
from dask.distributed import Client, LocalCluster
import dask.bag as db
import glob
import argparse
import pandas as pd


def process_traj_wrapper(traj_path):
    return ProcessValTraj(traj_path).get_result()


def get_bulk_elements_and_num(stoichiometry):
    """Get the unique bulk elements and number of from stoichiometry"""
    return list(stoichiometry.keys()), len(list(stoichiometry.keys()))


if __name__ == "__main__":
    # Set things up to parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--folders", nargs="*", type=str)
    arg_parser.add_argument(
        "--dft_df_path", type=str, default="catlas/parity/df_pkls/OC_20_val_data.pkl"
    )
    arg_parser.add_argument("--model_id", type=str)

    arg_parser.add_argument("--n_workers", type=int, default=4)

    # parse the command line and unpack them
    args = arg_parser.parse_args()
    folders = args.folders
    dft_df_path = args.dft_df_path
    model_id = args.model_id
    n_workers = args.n_workers

    # Start the cluster to parallelize processing
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)

    # Get all traj files
    files = []
    for folder in folders:
        files_now = glob.glob(folder + "*.traj")
        files.extend(files_now)

    # Process trajectories
    entries_bag = db.from_sequence(files).repartition(npartitions=20)
    results = entries_bag.map(process_traj_wrapper).compute()

    # Move to df and add metadata and dft
    ml_df = pd.DataFrame(results)
    dft_df = pd.read_pickle(dft_df_path)
    merged_df = ml_df.merge(dft_df, right_on="random_id", left_on="random_id")

    # Use stoichiometry to grab bulk elements and number of
    merged_df["bulk_elements"], merged_df["bulk_nelements"] = zip(
        *merged_df.stoichiometry.apply(get_bulk_elements_and_num)
    )

    merged_df = merged_df[merged_df.status]

    # Write df to pkl for future use in parity
    merged_df.to_pickle("catlas/parity/df_pkls/" + model_id + ".pkl")
