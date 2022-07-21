"""Create validation pickle files from npz predictions on val."""

from catlas.parity.data_processing_utils import ProcessValNPZ
import argparse

if __name__ == "__main__":
    # Parse arguments:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--npz_path", type=str)
    arg_parser.add_argument(
        "--dft_df_path", type=str, default="catlas/parity/df_pkls/OC_20_val_data.pkl"
    )
    args = arg_parser.parse_args()
    npz_path = args.npz_path
    dft_df_path = args.dft_df_path

    # Process NPZ
    ProcessValNPZ(npz_path, dft_df_path).process_data()
