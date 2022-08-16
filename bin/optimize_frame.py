import yaml
from catlas.config_validation import config_validator
import sys
from jinja2 import Template
import os
import time
import warnings
import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
from catlas.filters import bulk_filter
from catlas.parity.parity_utils import make_parity_plots
import numpy as np


def get_errors(row):
    return list(np.abs(np.array(row.ML_energy) - row.DFT_energy)), list(
        np.array(row.ML_energy) - row.DFT_energy
    )


# Load inputs and define global vars
if __name__ == "__main__":
    """The number of relaxation steps is set arbitrarily. This script determines the
        optimal number of steps using OCP validation data.

    Raises:
        ValueError: The provided config is invalid.
        ValueError: The model does not appear to run relaxations, so no frame
            optimization is necessary.
    """
    # Load the config yaml
    config_path = sys.argv[1]
    template = Template(open(config_path).read())
    config = yaml.load(template.render(**os.environ), Loader=yaml.FullLoader)
    if not config_validator.validate(config):
        raise ValueError(
            "Config has the following errors:\n%s"
            % "\n".join(
                [
                    ": ".join(['"%s"' % str(i) for i in item])
                    for item in config_validator.errors.items()
                ]
            )
        )
    else:
        print("Config validated")

    # Load the appropriate parity pkl file and filter it
    model_id = (
        config["adslab_prediction_steps"][0]["checkpoint_path"]
        .split("/")[-1]
        .split(".")[0]
    )
    df = pd.read_pickle("catlas/parity/df_pkls/" + model_id + ".pkl")
    ddf = dd.from_pandas(df, npartitions=1)
    df_filtered = bulk_filter(config, ddf).compute()

    # Raise an error if model doesnt have relaxation data
    if type(df.ML_energy.tolist()[0]) != list:
        raise ValueError(
            "Dataset is not relaxation like, there is no frame to optimise"
        )

    # warn that the first model will be used if there are more than one
    if len(config["adslab_prediction_steps"]) > 1:
        warnings.warn(
            f"""Multiple models were specified, the first one ({model_id}) will be
                optimised"""
        )

    frame_output = []
    adsorbates = config["adsorbate_filters"]["filter_by_smiles"]
    for adsorbate in adsorbates:
        df_smile = df_filtered[df_filtered.adsorbate == adsorbate]
        df_smile["per_frame_abs_errors"], df_smile["per_frame_errors"] = zip(
            *df_smile.apply(get_errors, axis=1)
        )
        smile_errors = np.array(df_smile.per_frame_errors.tolist())
        smile_abs_errors = np.array(df_smile.per_frame_abs_errors.tolist())
        maes = []
        mes = []
        for idx, _ in enumerate(smile_errors[0]):
            maes.append(sum(smile_abs_errors[:, idx]) / np.shape(smile_abs_errors)[0])
            mes.append(sum(smile_errors[:, idx]) / np.shape(smile_errors)[0])

        # Save info about the best frame
        output_now = {
            "adsorbate": adsorbate,
            "MAE": min(maes),
            "best_frame": maes.index(min(maes)),
            "per_frame_mae": maes,
            "per_frame_me": mes,
        }
        frame_output.append(output_now)

    # Make a folder for outputs
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + model_id + "-frame_opt"
    os.makedirs(f"outputs/{run_id}/")

    # Print and pickle a summary df
    df = pd.DataFrame(frame_output)
    print(df)
    df.to_pickle(f"outputs/{run_id}/" + "summary.pkl")

    # Make a plot of the maes v frame
    f, (ax1, ax2) = plt.subplots(1, 2)
    mes_all = df.per_frame_me.tolist()
    best_frame = df.best_frame.tolist()
    frame_by_smile = {}
    for idx, mae_set in enumerate(df.per_frame_mae.tolist()):
        frame_by_smile[adsorbates[idx]] = best_frame[idx]
        ax1.plot(range(len(mae_set)), mae_set, label=adsorbates[idx])
        ax2.plot(range(len(mes_all[idx])), mes_all[idx], label=adsorbates[idx])
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel("step number")
    ax1.set_ylabel("Mean Absolute Error")
    ax2.set_xlabel("step number")
    ax2.set_ylabel("Mean Error")
    f.set_figwidth(12)
    f.savefig(f"outputs/{run_id}/" + "mae_v_frame.pdf")

    # Make parity plot
    make_parity_plots(df_filtered, config, f"outputs/{run_id}/", frame_by_smile)
