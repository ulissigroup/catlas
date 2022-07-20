"""Create validation pickle files from npz predictions on val."""

from catlas.parity.data_processing_utils import ProcessValNPZ
import sys

if __name__ == "__main__":
    npz_path = sys.argv[1]
    dft_df_path = sys.argv[2]
    ProcessValNPZ(npz_path, dft_df_path).process_data()
