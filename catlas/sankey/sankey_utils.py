import plotly.graph_objects as go
import warnings
import numpy as np


def update_dictionary(
    sankey_dict: dict, label: str, source: int, target: int, value: int
) -> dict:
    """
    Updates the Sankey dictionary with a new edge (flow of content).

    Args:
        sankey_dict: a dictionary of values that will be used to populate the output sankey diagram
        label: a new label to add to the labels
        source: the node from which the flow comes
        target: the node to which the flow goes
        value: the magnitude of the flow

    Returns:
        sankey_dict: the sankey dictionary with new info added
    """
    if label is not None:
        sankey_dict["label"].append(label)
    sankey_dict["source"].append(source)
    sankey_dict["target"].append(target)
    sankey_dict["value"].append(value)
    return sankey_dict


def add_slab_info(sankey_dict: dict, num_unfiltered: int, num_filtered: int) -> dict:
    """
    Updates the Sankey dictionary with slab information.

    Args:
        sankey_dict: a dictionary of values that will be used to populate the output sankey diagram
        num_unfiltered: the number of slabs prior to filtering
        num_filtered: the number of slabs after filtering

    Returns:
        sankey_dict: the sankey dictionary with slab info added
    """
    node_idx = len(sankey_dict["label"])
    slab_idx = sankey_dict["label"].index("Slabs")
    sankey_dict = update_dictionary(
        sankey_dict,
        f"Filtered slabs ({num_filtered})",
        slab_idx,
        node_idx,
        num_filtered,
    )
    sankey_dict = update_dictionary(
        sankey_dict,
        f"Rejected slabs ({num_unfiltered - num_filtered})",
        slab_idx,
        node_idx + 1,
        num_unfiltered - num_filtered,
    )
    sankey_dict = update_dictionary(
        sankey_dict, None, node_idx, sankey_dict["label"].index("Adslabs"), num_filtered
    )
    sankey_dict["label"][slab_idx] = f"Slabs ({num_unfiltered})"
    return sankey_dict


def add_adslab_info(sankey_dict: dict, num_adslabs: int, num_inference: int) -> dict:
    """
    Updates the Sankey dictionary with adslab information.

    Args:
        sankey_dict: a dictionary of values that will be used to populate the output sankey diagram
        num_adslabs: the number of adslabs
        num_inference: the number of inference calculations

    Returns:
        sankey_dict: the sankey dictionary with adslab info added
    """
    if num_adslabs is None:
        warnings.warn(
            "Adslabs were not computed and therefore will not appear in the Sankey diagram"
        )
        num_adslabs = 0
    adslab_idx = sankey_dict["label"].index("Adslabs")
    sankey_dict = update_dictionary(
        sankey_dict,
        f"Inferred energies ({num_inference})",
        adslab_idx,
        len(sankey_dict["label"]),
        num_inference,
    )
    sankey_dict["label"][adslab_idx] = f"Adslabs ({num_adslabs})"
    if num_adslabs != num_inference:
        sankey_dict = update_dictionary(
            sankey_dict,
            f"Not inferred adslabs ({num_adslabs - num_inference})",
            adslab_idx,
            len(sankey_dict["label"]),
            num_adslabs - num_inference,
        )
    return sankey_dict


def get_sankey_diagram(sankey_dict: dict, run_id: str, use_log=True):
    """
    A function to create a pdf of the Sankey diagram.

    Args:
        sankey_dict: a dictionary of values that will be used to populate the output sankey diagram
        run_id: unique id for the run to be used as a location for saving outputs

    Returns:
        a pdf of the sankey diagram for the run
    """
    if use_log:
        vals = np.log(sankey_dict["value"])
        values = [i if i > 0 else 0.1 for i in vals]
    else:
        values = sankey_dict["value"]
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=30,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=sankey_dict["label"],
                ),
                link=dict(
                    source=sankey_dict["source"],
                    target=sankey_dict["target"],
                    value=values,
                ),
            )
        ]
    )
    fig.update_layout(
        autosize=False,
        width=1600,
        height=800,
    )
    fig.write_image(f"outputs/{run_id}/sankey.png")
