import plotly.graph_objects as go


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
    slab_idx = sankey_dict["label"].index("Slabs")
    sankey_dict = update_dictionary(sankey_dict, f"Filtered slabs ({num_filtered})", slab_idx, len(sankey_dict["label"]), num_filtered)
    sankey_dict = update_dictionary(sankey_dict, f"Rejected slabs ({num_unfiltered - num_filtered})", slab_idx, len(sankey_dict["label"]), num_unfiltered - num_filtered)
    sankey_dict["label"][slab_idx] = f"Slabs ({num_unfiltered})"
    return sankey_dict

def add_adslab_info(sankey_dict: dict, number: int) -> dict:
    """
    Updates the Sankey dictionary with adslab information.

    Args:
        sankey_dict: a dictionary of values that will be used to populate the output sankey diagram
        number: the number of adslabs

    Returns:
        sankey_dict: the sankey dictionary with adslab info added
    """
    adslab_idx = sankey_dict["label"].index("Adslabs")
    sankey_dict = update_dictionary(sankey_dict, f"Inferred energies ({number})", slab_idx, len(sankey_dict["label"]), number)
    sankey_dict["label"][adslab_idx] = f"Adslabs ({number})"
    return sankey_dict

def get_sankey_diagram(sankey_dict: dict, run_id: str):
    """
    A function to create a pdf of the Sankey diagram.

    Args:
        sankey_dict: a dictionary of values that will be used to populate the output sankey diagram
        run_id: unique id for the run to be used as a location for saving outputs

    Returns:
        a pdf of the sankey diagram for the run
    """
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=sankey_dict["label"],
                ),
                link=dict(
                    source=sankey_dict["source"],
                    target=sankey_dict["target"],
                    value=sankey_dict["value"],
                ),
            )
        ]
    )
    fig.write_image(f"outputs/{run_id}/sankey.png")
    
