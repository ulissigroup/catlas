from parity_utils import (
    get_predicted_E,
    data_preprocessing,
    apply_filters,
    get_specific_smile_plot,
    get_general_plot,
)
import yaml
import sys
import pandas as pd
import datetime

if __name__ == "__main__":

    # Load the config yaml
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load the data
    df = data_preprocessing(config["npz_file_path"], config["dft_data_path"])

    # Apply filters
    df_filtered = apply_filters(config, df)

    list_of_parity_info = []
    # Generate adsorbate specific plots
    for smile in config["desired_adsorbate_smiles"]:
        info_now = get_specific_smile_plot(smile, df_filtered, config["npz_file_path"])
        list_of_parity_info.append(info_now)

    # Generate overall model plot
    info_now = get_general_plot(df_filtered, config["npz_file_path"])
    list_of_parity_info.append(info_now)

    # Create a pickle of the summary info and print results
    df = pd.DataFrame(list_of_parity_info)
    print(df)
    model_id = config["npz_file_path"].split("/")[-1]
    model_id = model_id.split(".")[0]
    time_now = str(datetime.datetime.now())
    df_file_path = "output_pkls/df_summary_" + model_id + time_now + ".pkl"
    df.to_pickle(df_file_path)
