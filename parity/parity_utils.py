from numpy import load
import numpy as np
import pickle
import pandas as pd
import re
from os.path import exists
import warnings
import os
import matplotlib.pyplot as plt
import datetime
from scipy.stats import linregress


def get_predicted_E(row, ML_data):

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
    model_id = npz_path.split("/")[-1]
    model_id = model_id.split(".")[0]

    df_path = "df_pkls/" + model_id + ".pkl"
    if not exists(df_path):

        # Open files
        ML_data = dict(load(npz_path))
        with open("df_pkls/OC_20_val_data.pkl", "rb") as f:
            dft_df = pd.read_pickle(f)

        # Get ML energies
        dft_df["ML_energy"] = dft_df.apply(get_predicted_E, args=(ML_data,), axis=1)

        # Pickle results for future use
        dft_df.to_pickle(df_path)

    else:
        dft_df = pd.read_pickle(df_path)

    return dft_df


def apply_filters(config: dict, df: pd.DataFrame) -> pd.DataFrame:
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

    element_filters = config["element_filters"]

    for name, val in element_filters.items():
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

            elif name == "filter_by_number_elements":
                df["filter_number_els"] = df.stoichiometry.apply(
                    get_number_elements_boolean, args=(val,)
                )
                df = df[df.filter_number_els]
                df = df.drop(columns=["filter_number_els"])

            else:
                warnings.warn(name + " has not been implemented")
    return df


def get_specific_smile_plot(smile: str, df: pd.DataFrame, npz_path: str):

    # Create directory if it doesnt exist
    model_id = npz_path.split("/")[-1]
    model_id = model_id.split(".")[0]

    file_path = "output-plots/" + model_id
    if not exists(file_path):
        os.mkdir(file_path)

    # Create the plot if one doesnt already exist
    time_now = str(datetime.datetime.now())
    plot_file_path = file_path + "/" + time_now + "_" + smile + ".pdf"

    df_smile_specific = df[df.adsorbate == smile]

    if len(df_smile_specific.distribution.tolist()) == 0:
        warnings.warn(smile + "does not appear in the validation split")
    else:
        types = np.unique(df_smile_specific.distribution.tolist())

        x_overall = df_smile_specific["energy dE [eV]"].tolist()
        y_overall = df_smile_specific.ML_energy.tolist()
        MAE_overall = sum(abs(np.array(x_overall) - np.array(y_overall))) / len(
            x_overall
        )
        slope_overall, intercept_overall, r_overall, p, se = linregress(
            x_overall, y_overall
        )
        f, (ax1, ax2, ax3) = plt.subplots(1, len(types) + 1, sharey=True)
        ax1.set_title(smile + " overall")
        ax1.plot([-4, 2], [-4, 2], "k-", linewidth=3)
        ax1.plot(
            [-4, 2],
            [
                -4 * slope_overall + intercept_overall,
                2 * slope_overall + intercept_overall,
            ],
            "k--",
            linewidth=2,
        )
        ax1.scatter(x_overall, y_overall, s=6, facecolors="none", edgecolors="b")
        ax1.text(-3.95, 1.75, f"MAE = {MAE_overall:1.2f} eV")
        ax1.text(-3.95, 1.4, f"N points = {len(x_overall)}")
        ax1.legend(
            [
                "y = x",
                f"y = {slope_overall:1.2f} x + {intercept_overall:1.2f}, R-sq = {r_overall**2:1.2f}",
            ],
            loc="lower right",
        )
        ax1.axis("square")
        ax1.set_xlim([-4, 2])
        ax1.set_ylim([-4, 2])
        ax1.set_xticks([-4, -3, -2, -1, 0, 1, 2])
        ax1.set_yticks([-4, -3, -2, -1, 0, 1, 2])
        ax1.set_xlabel("DFT adsorption E [eV]")
        ax1.set_ylabel("ML adsorption E [eV]")

        df_now = df_smile_specific[df_smile_specific.distribution == types[0]]
        x_now = df_now["energy dE [eV]"].tolist()
        y_now = df_now.ML_energy.tolist()
        MAE_now = sum(abs(np.array(x_now) - np.array(y_now))) / len(x_now)
        slope_now, intercept_now, r_now, p, se = linregress(x_now, y_now)

        ax2.set_title(smile + " " + str(types[0]))
        ax2.plot([-4, 2], [-4, 2], "k-", linewidth=3)
        ax2.plot(
            [-4, 2],
            [-4 * slope_now + intercept_now, 2 * slope_now + intercept_now],
            "k--",
            linewidth=2,
        )
        ax2.scatter(x_now, y_now, s=6, facecolors="none", edgecolors="b")
        ax2.text(-3.95, 1.75, f"MAE = {MAE_now:1.2f} eV")
        ax2.text(-3.95, 1.4, f"N points = {len(x_now)}")
        ax2.legend(
            [
                "y = x",
                f"y = {slope_now:1.2f} x + {intercept_now:1.2f}, R-sq = {r_now**2:1.2f}",
            ],
            loc="lower right",
        )
        ax2.axis("square")
        ax2.set_xlim([-4, 2])
        ax2.set_ylim([-4, 2])
        ax2.set_xticks([-4, -3, -2, -1, 0, 1, 2])
        ax2.set_yticks([-4, -3, -2, -1, 0, 1, 2])
        ax2.set_xlabel("DFT adsorption E [eV]")
        ax2.set_ylabel("ML adsorption E [eV]")

        df_now = df_smile_specific[df_smile_specific.distribution == types[1]]
        x_now = df_now["energy dE [eV]"].tolist()
        y_now = df_now.ML_energy.tolist()
        MAE_now = sum(abs(np.array(x_now) - np.array(y_now))) / len(x_now)
        slope_now, intercept_now, r_now, p, se = linregress(x_now, y_now)

        ax3.set_title(smile + " " + str(types[1]))
        ax3.plot([-4, 2], [-4, 2], "k-", linewidth=3)
        ax3.plot(
            [-4, 2],
            [-4 * slope_now + intercept_now, 2 * slope_now + intercept_now],
            "k--",
            linewidth=2,
        )
        ax3.scatter(x_now, y_now, s=6, facecolors="none", edgecolors="b")
        ax3.text(-3.95, 1.75, f"MAE = {MAE_now:1.2f} eV")
        ax3.text(-3.95, 1.4, f"N points = {len(x_now)}")
        ax3.legend(
            [
                "y = x",
                f"y = {slope_now:1.2f} x + {intercept_now:1.2f}, R-sq = {r_now**2:1.2f}",
            ],
            loc="lower right",
        )
        ax3.axis("square")
        ax3.set_xlim([-4, 2])
        ax3.set_ylim([-4, 2])
        ax3.set_xticks([-4, -3, -2, -1, 0, 1, 2])
        ax3.set_yticks([-4, -3, -2, -1, 0, 1, 2])
        ax3.set_xlabel("DFT adsorption E [eV]")
        ax3.set_ylabel("ML adsorption E [eV]")

        f.set_figwidth(18)
        f.savefig(plot_file_path)


def get_general_plot(df: pd.DataFrame, npz_path: str):

    # Create directory if it doesnt exist
    model_id = npz_path.split("/")[-1]
    model_id = model_id.split(".")[0]

    file_path = "output-plots/" + model_id
    if not exists(file_path):
        os.mkdir(file_path)

    # Create the plot if one doesnt already exist
    time_now = str(datetime.datetime.now())
    plot_file_path = file_path + "/" + time_now + "_" + "general" + ".pdf"

    types = np.unique(df.distribution.tolist())

    x_overall = df["energy dE [eV]"].tolist()
    y_overall = df.ML_energy.tolist()
    MAE_overall = sum(abs(np.array(x_overall) - np.array(y_overall))) / len(x_overall)
    slope_overall, intercept_overall, r_overall, p, se = linregress(
        x_overall, y_overall
    )
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True)
    ax1.set_title("overall")
    ax1.plot([-4, 2], [-4, 2], "k-", linewidth=3)
    ax1.plot(
        [-4, 2],
        [-4 * slope_overall + intercept_overall, 2 * slope_overall + intercept_overall],
        "k--",
        linewidth=2,
    )
    ax1.scatter(x_overall, y_overall, s=6, facecolors="none", edgecolors="b")
    ax1.text(-3.95, 1.75, f"MAE = {MAE_overall:1.2f} eV")
    ax1.text(-3.95, 1.4, f"N points = {len(x_overall)}")
    ax1.legend(
        [
            "y = x",
            f"y = {slope_overall:1.2f} x + {intercept_overall:1.2f}, R-sq = {r_overall**2:1.2f}",
        ],
        loc="lower right",
    )
    ax1.axis("square")
    ax1.set_xlim([-4, 2])
    ax1.set_ylim([-4, 2])
    ax1.set_xticks([-4, -3, -2, -1, 0, 1, 2])
    ax1.set_yticks([-4, -3, -2, -1, 0, 1, 2])
    ax1.set_xlabel("DFT adsorption E [eV]")
    ax1.set_ylabel("ML adsorption E [eV]")

    df_now = df[df.distribution == types[0]]
    x_now = df_now["energy dE [eV]"].tolist()
    y_now = df_now.ML_energy.tolist()
    MAE_now = sum(abs(np.array(x_now) - np.array(y_now))) / len(x_now)
    slope_now, intercept_now, r_now, p, se = linregress(x_now, y_now)

    ax2.set_title(str(types[0]))
    ax2.plot([-4, 2], [-4, 2], "k-", linewidth=3)
    ax2.plot(
        [-4, 2],
        [-4 * slope_now + intercept_now, 2 * slope_now + intercept_now],
        "k--",
        linewidth=2,
    )
    ax2.scatter(x_now, y_now, s=6, facecolors="none", edgecolors="b")
    ax2.text(-3.95, 1.75, f"MAE = {MAE_now:1.2f} eV")
    ax2.text(-3.95, 1.4, f"N points = {len(x_now)}")
    ax2.legend(
        [
            "y = x",
            f"y = {slope_now:1.2f} x + {intercept_now:1.2f}, R-sq = {r_now**2:1.2f}",
        ],
        loc="lower right",
    )
    ax2.axis("square")
    ax2.set_xlim([-4, 2])
    ax2.set_ylim([-4, 2])
    ax2.set_xticks([-4, -3, -2, -1, 0, 1, 2])
    ax2.set_yticks([-4, -3, -2, -1, 0, 1, 2])
    ax2.set_xlabel("DFT adsorption E [eV]")
    ax2.set_ylabel("ML adsorption E [eV]")

    df_now = df[df.distribution == types[1]]
    x_now = df_now["energy dE [eV]"].tolist()
    y_now = df_now.ML_energy.tolist()
    MAE_now = sum(abs(np.array(x_now) - np.array(y_now))) / len(x_now)
    slope_now, intercept_now, r_now, p, se = linregress(x_now, y_now)

    ax3.set_title(str(types[1]))
    ax3.plot([-4, 2], [-4, 2], "k-", linewidth=3)
    ax3.plot(
        [-4, 2],
        [-4 * slope_now + intercept_now, 2 * slope_now + intercept_now],
        "k--",
        linewidth=2,
    )
    ax3.scatter(x_now, y_now, s=6, facecolors="none", edgecolors="b")
    ax3.text(-3.95, 1.75, f"MAE = {MAE_now:1.2f} eV")
    ax3.text(-3.95, 1.4, f"N points = {len(x_now)}")
    ax3.legend(
        [
            "y = x",
            f"y = {slope_now:1.2f} x + {intercept_now:1.2f}, R-sq = {r_now**2:1.2f}",
        ],
        loc="lower right",
    )
    ax3.axis("square")
    ax3.set_xlim([-4, 2])
    ax3.set_ylim([-4, 2])
    ax3.set_xticks([-4, -3, -2, -1, 0, 1, 2])
    ax3.set_yticks([-4, -3, -2, -1, 0, 1, 2])
    ax3.set_xlabel("DFT adsorption E [eV]")
    ax3.set_ylabel("ML adsorption E [eV]")

    df_now = df[df.distribution == types[2]]
    x_now = df_now["energy dE [eV]"].tolist()
    y_now = df_now.ML_energy.tolist()
    MAE_now = sum(abs(np.array(x_now) - np.array(y_now))) / len(x_now)
    slope_now, intercept_now, r_now, p, se = linregress(x_now, y_now)

    ax4.set_title(str(types[2]))
    ax4.plot([-4, 2], [-4, 2], "k-", linewidth=3)
    ax4.plot(
        [-4, 2],
        [-4 * slope_now + intercept_now, 2 * slope_now + intercept_now],
        "k--",
        linewidth=2,
    )
    ax4.scatter(x_now, y_now, s=6, facecolors="none", edgecolors="b")
    ax4.text(-3.95, 1.75, f"MAE = {MAE_now:1.2f} eV")
    ax4.text(-3.95, 1.4, f"N points = {len(x_now)}")
    ax4.legend(
        [
            "y = x",
            f"y = {slope_now:1.2f} x + {intercept_now:1.2f}, R-sq = {r_now**2:1.2f}",
        ],
        loc="lower right",
    )
    ax4.axis("square")
    ax4.set_xlim([-4, 2])
    ax4.set_ylim([-4, 2])
    ax4.set_xticks([-4, -3, -2, -1, 0, 1, 2])
    ax4.set_yticks([-4, -3, -2, -1, 0, 1, 2])
    ax4.set_xlabel("DFT adsorption E [eV]")
    ax4.set_ylabel("ML adsorption E [eV]")

    df_now = df[df.distribution == types[3]]
    x_now = df_now["energy dE [eV]"].tolist()
    y_now = df_now.ML_energy.tolist()
    MAE_now = sum(abs(np.array(x_now) - np.array(y_now))) / len(x_now)
    slope_now, intercept_now, r_now, p, se = linregress(x_now, y_now)

    ax5.set_title(str(types[3]))
    ax5.plot([-4, 2], [-4, 2], "k-", linewidth=3)
    ax5.plot(
        [-4, 2],
        [-4 * slope_now + intercept_now, 2 * slope_now + intercept_now],
        "k--",
        linewidth=2,
    )
    ax5.scatter(x_now, y_now, s=6, facecolors="none", edgecolors="b")
    ax5.text(-3.95, 1.75, f"MAE = {MAE_now:1.2f} eV")
    ax5.text(-3.95, 1.4, f"N points = {len(x_now)}")
    ax5.legend(
        [
            "y = x",
            f"y = {slope_now:1.2f} x + {intercept_now:1.2f}, R-sq = {r_now**2:1.2f}",
        ],
        loc="lower right",
    )
    ax5.axis("square")
    ax5.set_xlim([-4, 2])
    ax5.set_ylim([-4, 2])
    ax5.set_xticks([-4, -3, -2, -1, 0, 1, 2])
    ax5.set_yticks([-4, -3, -2, -1, 0, 1, 2])
    ax5.set_xlabel("DFT adsorption E [eV]")
    ax5.set_ylabel("ML adsorption E [eV]")

    f.set_figwidth(30)
    f.savefig(plot_file_path)
