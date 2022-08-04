from numpy import load
import numpy as np
import pickle
import pandas as pd
import re
from os.path import exists
import warnings
import os
import matplotlib.pyplot as plt
import time
from scipy.stats import linregress
from catlas.filter_utils import get_elements_in_groups
from catlas.filters import bulk_filter
import dask.dataframe as dd


def get_model_id(checkpoint_path: str) -> str:
    """get the npz path for a specific checkpoint file"""
    pt_filename = checkpoint_path.split("/")[-1]
    model_id = pt_filename.split(".")[0]
    return model_id


def get_specific_smile_plot(
    smile: str,
    df: pd.DataFrame,
    output_path: str,
    number_steps,
    energy_key1="DFT_energy",
    energy_key2="ML_energy",
) -> dict:
    """Creates the pdf parity plot for a given smile and returns a dictionary summarizing plot results"""
    # Create the plot
    time_now = time.strftime("%Y%m%d-%H%M%S")
    plot_file_path = output_path + "/" + time_now + smile + ".pdf"

    # Filter the data to only include the desired smile
    df_smile_specific = df[df.adsorbate == smile]

    # Check to make sure some data exists
    if df_smile_specific.empty:
        warnings.warn("No matching validation data was found for " + smile)
        return {}
    else:
        # Initialize splits and output dictionary
        types = list(np.unique(df_smile_specific.distribution.tolist()))
        info_dict = {
            "adsorbate": smile,
            "overall_N": np.nan,
            "overall_MAE": np.nan,
            "overall_slope": np.nan,
            "overall_int": np.nan,
            "overall_r_sq": np.nan,
            "id_N": np.nan,
            "id_MAE": np.nan,
            "id_slope": np.nan,
            "id_int": np.nan,
            "id_r_sq": np.nan,
            "ood_N": np.nan,
            "ood_MAE": np.nan,
            "ood_slope": np.nan,
            "ood_int": np.nan,
            "ood_r_sq": np.nan,
            "ood_cat_N": np.nan,
            "ood_cat_MAE": np.nan,
            "ood_cat_slope": np.nan,
            "ood_cat_int": np.nan,
            "ood_cat_r_sq": np.nan,
            "ood_ads_N": np.nan,
            "ood_ads_MAE": np.nan,
            "ood_ads_slope": np.nan,
            "ood_ads_int": np.nan,
            "ood_ads_r_sq": np.nan,
        }

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

        # Process data for all splits
        info_dict_now = make_subplot(
            ax1, df_smile_specific, "overall", number_steps, energy_key1, energy_key2
        )
        info_dict = update_info(info_dict, "overall", info_dict_now)

        # Process data for split 1
        df_now = df_smile_specific[df_smile_specific.distribution == types[0]]
        name_now = smile + " " + types[0]
        info_dict_now = make_subplot(
            ax2, df_now, name_now, number_steps, energy_key1, energy_key2
        )
        info_dict = update_info(info_dict, name_now, info_dict_now)

        if len(types) == 2:
            # Process data for split 2 if it exists
            df_now = df_smile_specific[df_smile_specific.distribution == types[1]]
            name_now = smile + " " + types[1]
            info_dict_now = make_subplot(
                ax3, df_now, name_now, number_steps, energy_key1, energy_key2
            )
            info_dict = update_info(info_dict, name_now, info_dict_now)

        f.set_figwidth(18)
        f.savefig(plot_file_path)
        plt.close(f)
        return info_dict


def get_general_plot(
    df: pd.DataFrame,
    output_path: str,
    number_steps,
    energy_key1="DFT_energy",
    energy_key2="ML_energy",
) -> dict:
    """Creates the pdf parity plot for all smiles and returns a dictionary summarizing plot results"""
    # Check to make sure some data exists
    if df.empty:
        warnings.warn("No matching validation data was found")
        return {}
    else:
        info_dict = {
            "adsorbate": "all",
            "overall_N": np.nan,
            "overall_MAE": np.nan,
            "overall_slope": np.nan,
            "overall_int": np.nan,
            "overall_r_sq": np.nan,
            "id_N": np.nan,
            "id_MAE": np.nan,
            "id_slope": np.nan,
            "id_int": np.nan,
            "id_r_sq": np.nan,
            "ood_N": np.nan,
            "ood_MAE": np.nan,
            "ood_slope": np.nan,
            "ood_int": np.nan,
            "ood_r_sq": np.nan,
            "ood_cat_N": np.nan,
            "ood_cat_MAE": np.nan,
            "ood_cat_slope": np.nan,
            "ood_cat_int": np.nan,
            "ood_cat_r_sq": np.nan,
            "ood_ads_N": np.nan,
            "ood_ads_MAE": np.nan,
            "ood_ads_slope": np.nan,
            "ood_ads_int": np.nan,
            "ood_ads_r_sq": np.nan,
        }

        # Create the plot
        time_now = time.strftime("%Y%m%d-%H%M%S")
        plot_file_path = output_path + "/" + time_now + "_" + "general" + ".pdf"

        types = np.unique(df.distribution.tolist())

        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True)

        # Process data for all splits
        info_dict_now = make_subplot(
            ax1, df, "overall", number_steps, energy_key1, energy_key2
        )
        info_dict = update_info(info_dict, "overall", info_dict_now)

        # Process data for split 1
        df_now = df[df.distribution == types[0]]
        name_now = types[0]
        info_dict_now = make_subplot(
            ax2, df_now, name_now, number_steps, energy_key1, energy_key2
        )
        info_dict = update_info(info_dict, name_now, info_dict_now)

        # Process data for split 2 if it exists
        if len(types) >= 2:
            df_now = df[df.distribution == types[1]]
            name_now = types[1]
            info_dict_now = make_subplot(
                ax3, df_now, name_now, number_steps, energy_key1, energy_key2
            )
            info_dict = update_info(info_dict, name_now, info_dict_now)

        # Process data for split 3 if it exists
        if len(types) >= 3:
            df_now = df[df.distribution == types[2]]
            name_now = types[2]
            info_dict_now = make_subplot(
                ax4, df_now, name_now, number_steps, energy_key1, energy_key2
            )
            info_dict = update_info(info_dict, name_now, info_dict_now)

        # Process data for split 4 if it exists
        if len(types) == 4:
            df_now = df[df.distribution == types[3]]
            name_now = types[3]
            info_dict_now = make_subplot(
                ax5, df_now, name_now, number_steps, energy_key1, energy_key2
            )
            info_dict = update_info(info_dict, name_now, info_dict_now)

        f.set_figwidth(30)
        f.savefig(plot_file_path)
        plt.close(f)

        return info_dict


def make_subplot(subplot, df, name, number_steps, energy_key1, energy_key2) -> dict:
    """Helper function for larger plot generation. Processes each subplot."""
    x = np.array(df[energy_key1].tolist())
    y = np.array(df[energy_key2].tolist())
    if len(np.shape(y)) == 2:
        y = y[:, number_steps]
    MAE = np.sum(np.abs(x - y)) / len(x)
    slope, intercept, r, p, se = linregress(x, y)

    subplot.set_title(name)
    subplot.plot([-4, 2], [-4, 2], "k-", linewidth=3)
    subplot.plot(
        [-4, 2],
        [
            -4 * slope + intercept,
            2 * slope + intercept,
        ],
        "k--",
        linewidth=2,
    )
    subplot.scatter(x, y, s=6, facecolors="none", edgecolors="b")
    subplot.text(-3.95, 1.75, f"MAE = {MAE:1.2f} eV")
    subplot.text(-3.95, 1.4, f"N points = {len(x)}")
    subplot.legend(
        [
            "y = x",
            f"y = {slope:1.2f} x + {intercept:1.2f}, R-sq = {r**2:1.2f}",
        ],
        loc="lower right",
    )
    subplot.axis("square")
    subplot.set_xlim([-4, 2])
    subplot.set_ylim([-4, 2])
    subplot.set_xticks([-4, -3, -2, -1, 0, 1, 2])
    subplot.set_yticks([-4, -3, -2, -1, 0, 1, 2])
    subplot.set_xlabel(energy_key1 + " [eV]")
    subplot.set_ylabel(energy_key2 + " [eV]")
    return {"N": len(x), "MAE": MAE, "slope": slope, "intercept": intercept, "r": r}


def update_info(info_dict: dict, name: str, info_to_add: dict) -> dict:
    """Helper function for summary dictionary generation. Updates the dictionary for each split."""
    info_dict[name + "_N"] = info_to_add["N"]
    info_dict[name + "_MAE"] = info_to_add["MAE"]
    info_dict[name + "_slope"] = info_to_add["slope"]
    info_dict[name + "_int"] = info_to_add["intercept"]
    info_dict[name + "_r_sq"] = info_to_add["r"] ** 2
    return info_dict


def get_parity_upfront(config, run_id):
    """
    Get parity plot to cover intended scope of work in catlas.

    Args:
        config: a dictionary loaded from a catlas input yaml
        run_id: name of the output folder
    """

    if "adslab_prediction_steps" in config:

        ## Create an output folder
        if not os.path.exists(f"outputs/{run_id}/parity/"):
            os.makedirs(f"outputs/{run_id}/parity/")

        ## Iterate over steps
        inference_steps = [
            step
            for step in config["adslab_prediction_steps"]
            if step.get("type", "inference")
        ]
        for step in inference_steps:
            ### Load the data
            model_id = get_model_id(step["checkpoint_path"])
            if os.path.exists("catlas/parity/df_pkls/" + model_id + ".pkl"):
                number_steps = step["number_steps"] if "number_steps" in step else 200

                ### Apply filters
                df = pd.read_pickle("catlas/parity/df_pkls/" + model_id + ".pkl")
                ddf = dd.from_pandas(df, npartitions=2)
                df_filtered = bulk_filter(config, ddf).compute()

                ### Generate a folder for each model to be considered
                folder_now = f"outputs/{run_id}/parity/" + step["label"]
                if not os.path.exists(folder_now):
                    os.makedirs(folder_now)
                make_parity_plots(df_filtered, config, folder_now, number_steps)

            else:
                warnings.warn(
                    model_id
                    + " validation pkl has not been found and therefore parity plots cannot be generated"
                )


def make_parity_plots(df_filtered, config, output_path, number_steps_all):
    list_of_parity_info = []
    # Generate adsorbate specific plots
    for smile in config["adsorbate_filters"]["filter_by_smiles"]:
        ## Parse specific step numbers where applicable
        if type(number_steps_all) == dict:
            number_steps = number_steps_all[smile]
        else:
            number_steps = number_steps_all
        info_now = get_specific_smile_plot(
            smile, df_filtered, output_path, number_steps
        )
        list_of_parity_info.append(info_now)

    # Generate overall model plot
    info_now = get_general_plot(df_filtered, output_path, number_steps=200)
    list_of_parity_info.append(info_now)

    # Create a pickle of the summary info and print results
    df = pd.DataFrame(list_of_parity_info)
    df_file_path = output_path + "parity_summary_df" + ".pkl"
    df.to_pickle(df_file_path)
