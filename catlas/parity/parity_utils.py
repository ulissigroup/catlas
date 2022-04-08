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


def get_npz_path(checkpoint_path: str) -> str:
    """get the npz path for a specific checkpoint file"""
    pt_filename = checkpoint_path.split("/")[-1]
    model_id = pt_filename.split(".")[0]
    return "parity/npz-files/" + model_id + ".npz"


def get_predicted_E(row, ML_data):
    """Finds the corresponding ML energy for a given DFT calculation"""
    random_id = row.random_id
    distribution = row.distribution
    id_now = re.findall(r"([0-9]+)", random_id)[0]
    if distribution == "id":
        idx = np.where(ML_data["id_ids"] == id_now)
        energy = ML_data["id_energy"][idx]
    elif distribution == "ood_ads":
        idx = np.where(ML_data["ood_ads_ids"] == id_now)
        energy = ML_data["ood_ads_energy"][idx]
    elif distribution == "ood_cat":
        idx = np.where(ML_data["ood_cat_ids"] == id_now)
        energy = ML_data["ood_cat_energy"][idx]
    elif distribution == "ood":
        idx = np.where(ML_data["ood_both_ids"] == id_now)
        energy = ML_data["ood_both_energy"][idx]
    return energy[0]


def data_preprocessing(npz_path: str, dft_df_path: str) -> pd.DataFrame:
    """Creates the primary dataframe for use in analysis"""
    model_id = npz_path.split("/")[-1]
    model_id = model_id.split(".")[0]

    df_path = "parity/df_pkls/" + model_id + ".pkl"
    if not exists(df_path):

        # Open files
        ML_data = dict(load(npz_path))
        with open(dft_df_path, "rb") as f:
            dft_df = pd.read_pickle(f)

        # Get ML energies
        dft_df["ML_energy"] = dft_df.apply(get_predicted_E, args=(ML_data,), axis=1)
        dft_df.rename(columns={"energy dE [eV]": "DFT_energy"}, inplace=True)

        # Pickle results for future use
        dft_df.to_pickle(df_path)

    else:
        dft_df = pd.read_pickle(df_path)

    return dft_df


def apply_filters(bulk_filters: dict, df: pd.DataFrame) -> pd.DataFrame:
    """filters the dataframe to only include material types specified in the yaml"""

    def get_acceptable_elements_boolean(
        stoichiometry: dict, acceptable_els: list
    ) -> bool:
        elements = set(stoichiometry.keys())
        return elements.issubset(acceptable_els)

    def get_required_elements_boolean(stoichiometry: dict, required_els: list) -> bool:
        elements = list(stoichiometry.keys())
        return all([required_el in elements for required_el in required_els])

    def get_number_elements_boolean(stoichiometry: dict, number_els: list) -> bool:
        element_num = len(list(stoichiometry.keys()))
        return element_num in number_els

    def get_active_host_boolean(stoichiometry: dict, active_host_els: dict) -> bool:
        active = active_host_els["active"]
        host = active_host_els["host"]
        elements = set(stoichiometry.keys())
        return all(
            [
                all([el in [*active, *host] for el in elements]),
                any([el in host for el in elements]),
                any([el in active for el in elements]),
            ]
        )

    for name, val in bulk_filters.items():
        if (
            str(val) != "None"
        ):  # depending on how yaml is created, val may either be "None" or NoneType
            if name == "filter_by_acceptable_elements":
                df["filter_acceptable_els"] = df.stoichiometry.apply(
                    get_acceptable_elements_boolean, args=(val,)
                )
                df = df[df.filter_acceptable_els]
                df = df.drop(columns=["filter_acceptable_els"])
            elif name == "filter_by_required_elements":
                df["filter_required_els"] = df.stoichiometry.apply(
                    get_required_elements_boolean, args=(val,)
                )
                df = df[df.filter_required_els]
                df = df.drop(columns=["filter_required_els"])

            elif name == "filter_by_num_elements":
                df["filter_number_els"] = df.stoichiometry.apply(
                    get_number_elements_boolean, args=(val,)
                )
                df = df[df.filter_number_els]
                df = df.drop(columns=["filter_number_els"])

            elif name == "filter_by_element_groups":
                valid_els = get_elements_in_groups(val)
                df["filter_acceptable_els"] = df.stoichiometry.apply(
                    get_acceptable_elements_boolean, args=(valid_els,)
                )
                df = df[df.filter_acceptable_els]
                df = df.drop(columns=["filter_acceptable_els"])

            elif name == "filter_by_elements_active_host":
                df["filter_active_host_els"] = df.stoichiometry.apply(
                    get_active_host_boolean, args=(val,)
                )
                df = df[df.filter_active_host_els]
                df = df.drop(columns=["filter_active_host_els"])

            elif name == "filter_ignore_mpids":
                continue
            elif name == "filter_by_mpids":
                warnings.warn(name + " has not been implemented for parity generation")
            elif name == "filter_by_object_size":
                continue
            else:
                warnings.warn(name + " has not been implemented for parity generation")
    return df


def get_specific_smile_plot(
    smile: str,
    df: pd.DataFrame,
    output_path: str,
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
            ax1, df_smile_specific, "overall", energy_key1, energy_key2
        )
        info_dict = update_info(info_dict, "overall", info_dict_now)

        # Process data for split 1
        df_now = df_smile_specific[df_smile_specific.distribution == types[0]]
        name_now = smile + " " + types[0]
        info_dict_now = make_subplot(ax2, df_now, name_now, energy_key1, energy_key2)
        info_dict = update_info(info_dict, name_now, info_dict_now)

        if len(types) == 2:
            # Process data for split 2 if it exists
            df_now = df_smile_specific[df_smile_specific.distribution == types[1]]
            name_now = smile + " " + types[1]
            info_dict_now = make_subplot(
                ax3, df_now, name_now, energy_key1, energy_key2
            )
            info_dict = update_info(info_dict, name_now, info_dict_now)

        f.set_figwidth(18)
        f.savefig(plot_file_path)
        plt.close(f)
        return info_dict


def get_general_plot(
    df: pd.DataFrame,
    output_path: str,
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
        info_dict_now = make_subplot(ax1, df, "overall", energy_key1, energy_key2)
        info_dict = update_info(info_dict, "overall", info_dict_now)

        # Process data for split 1
        df_now = df[df.distribution == types[0]]
        name_now = types[0]
        info_dict_now = make_subplot(ax2, df_now, name_now, energy_key1, energy_key2)
        info_dict = update_info(info_dict, name_now, info_dict_now)

        # Process data for split 2 if it exists
        if len(types) >= 2:
            df_now = df[df.distribution == types[1]]
            name_now = types[1]
            info_dict_now = make_subplot(
                ax3, df_now, name_now, energy_key1, energy_key2
            )
            info_dict = update_info(info_dict, name_now, info_dict_now)

        # Process data for split 3 if it exists
        if len(types) >= 3:
            df_now = df[df.distribution == types[2]]
            name_now = types[2]
            info_dict_now = make_subplot(
                ax4, df_now, name_now, energy_key1, energy_key2
            )
            info_dict = update_info(info_dict, name_now, info_dict_now)

        # Process data for split 4 if it exists
        if len(types) == 4:
            df_now = df[df.distribution == types[3]]
            name_now = types[3]
            info_dict_now = make_subplot(
                ax5, df_now, name_now, energy_key1, energy_key2
            )
            info_dict = update_info(info_dict, name_now, info_dict_now)

        f.set_figwidth(30)
        f.savefig(plot_file_path)
        plt.close(f)

        return info_dict


def make_subplot(subplot, df, name, energy_key1, energy_key2) -> dict:
    """Helper function for larger plot generation. Processes each subplot."""
    x = df[energy_key1].tolist()
    y = df[energy_key2].tolist()
    MAE = sum(abs(np.array(x) - np.array(y))) / len(x)
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
    if "adslab_prediction_steps" in config:

        ## Create an output folder
        try:
            if not os.path.exists(config["output_options"]["parity_output_folder"]):
                os.makedirs(config["output_options"]["parity_output_folder"])
        except RuntimeError:
            print("A folder for parity results must be specified in the config yaml.")

        ## Iterate over steps
        for step in config["adslab_prediction_steps"]:
            ### Load the data
            npz_path = get_npz_path(step["checkpoint_path"])
            if os.path.exists(npz_path):
                df = data_preprocessing(npz_path, "parity/df_pkls/OC_20_val_data.pkl")

                ### Apply filters
                df_filtered = apply_filters(config["bulk_filters"], df)

                list_of_parity_info = []

                ### Generate a folder for each model to be considered
                folder_now = f"/outputs/{run_id}parity/step["label"]/"
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
                df_file_path = folder_now + "parity_summary_df" + ".pkl"
                df.to_pickle(df_file_path)
            else:
                warnings.warn(
                    npz_path
                    + " has not been found and therefore parity plots cannot be generated"
                )
