from parity_utils import (
    get_predicted_E,
    data_preprocessing,
    apply_filters,
    get_specific_smile_plot,
    get_general_plot,
    get_npz_path,
)
import yaml
import sys
import pandas as pd
import datetime
from itertools import combinations
from jinja2 import Template
import os
import warnings
import numpy as np


# Load inputs and define global vars
if __name__ == "__main__":

    # Load the config yaml
    config_path = sys.argv[1]
    template = Template(open(config_path).read())
    config = yaml.load(template.render(**os.environ))

    if "models_to_assess" in config:
        # Generate DFT v. ML parity plots
        ## Create an output folder
        try:
            if not os.path.exists(config["output_options"]["parity_output_folder"]):
                os.makedirs(config["output_options"]["parity_output_folder"])
        except RuntimeError:
            print("A folder for parity results must be specified in the config yaml.")

        ## Iterate over steps
        for model in config["models_to_assess"]["checkpoint_paths"]:
            ### Grab model id
            model_id = model.split("/")[-1].split(".")[0]

            ### Load the data
            npz_path = get_npz_path(model)
            if os.path.exists(npz_path):
                df = data_preprocessing(npz_path, "parity/df_pkls/OC_20_val_data.pkl")

                ### Apply filters
                df_filtered = apply_filters(config["bulk_filters"], df)

                list_of_parity_info = []

                ### Generate a folder for each model to be considered
                folder_now = (
                    config["output_options"]["parity_output_folder"] + "/" + model_id
                )
                if not os.path.exists(folder_now):
                    os.makedirs(folder_now)

                ### Generate adsorbate specific plots
                for smile in config["adsorbate_filters"]["filter_by_smiles"]:
                    info_now = get_specific_smile_plot(smile, df_filtered, folder_now)
                    list_of_parity_info.append(info_now)

                ### Generate overall model plot
                info_now = get_general_plot(df_filtered, folder_now)
                list_of_parity_info.append(info_now)

                ### Create a pickle of the summary info and print results
                df = pd.DataFrame(list_of_parity_info)
                time_now = str(datetime.datetime.now())
                df_file_path = folder_now + time_now + ".pkl"
                df.to_pickle(df_file_path)
            else:
                warnings.warn(
                    npz_path
                    + " has not been found and therefore parity plots cannot be generated"
                )

        # Generate ML v. ML parity plots
        ## Enumerate combos
        if config["models_to_assess"]["make_ML_v_ML_parity"]:
            if len(config["models_to_assess"]["checkpoint_paths"]) > 1:
                ML_model_combos = list(
                    combinations(config["models_to_assess"]["checkpoint_paths"], 2)
                )
                for combo in ML_model_combos:
                    npz_path0 = get_npz_path(combo[0])
                    npz_path1 = get_npz_path(combo[1])
                    df0 = data_preprocessing(
                        npz_path0, "parity/df_pkls/OC_20_val_data.pkl"
                    )
                    energy_key1 = "ML_energy_" + npz_path0.split("/")[-1].split(".")[0]
                    df0.rename(columns={"ML_energy": energy_key1}, inplace=True)
                    df1 = data_preprocessing(
                        npz_path1, "parity/df_pkls/OC_20_val_data.pkl"
                    )
                    df1.drop(df1.columns.difference(["ML_energy"]), 1, inplace=True)
                    energy_key2 = "ML_energy_" + npz_path1.split("/")[-1].split(".")[0]
                    df1.rename(columns={"ML_energy": energy_key2}, inplace=True)

                    df = pd.concat([df0, df1], axis=1, join="inner")

                    ### Generate a folder for each model to be considered
                    folder_now = (
                        config["output_options"]["parity_output_folder"]
                        + "/"
                        + energy_key1
                        + "_"
                        + energy_key2
                    )
                    if not os.path.exists(folder_now):
                        os.makedirs(folder_now)

                    df_filtered = apply_filters(config["bulk_filters"], df)

                    ### Make smile specific plots
                    for smile in config["adsorbate_filters"]["filter_by_smiles"]:
                        info_now = get_specific_smile_plot(
                            smile,
                            df_filtered,
                            folder_now,
                            energy_key1=energy_key1,
                            energy_key2=energy_key2,
                        )
                    ### Generate overall model plot
                    info_now = get_general_plot(
                        df_filtered,
                        folder_now,
                        energy_key1=energy_key1,
                        energy_key2=energy_key2,
                    )
            else:
                print(
                    "ML to ML comparison couldn't be made because not enough models were given."
                )

    print("Done making parity plots where data was available.")
