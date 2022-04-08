import plotly.graph_objects as go
    
    
def update_dictionary(sankey_dict: dict, label: str, source: int, target: int, value: int) -> dict:
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
        sankey_dict['label'].append(label)
    sankey_dict['source'].append(source)
    sankey_dict['target'].append(target)
    sankey_dict['value'].append(value)
    return sankey_dict
    
def get_sankey_diagram(sankey_dict: dict, config: dict):
    """
    A function to create a pdf of the Sankey diagram.
    
    Args:
        sankey_dict: a dictionary of values that will be used to populate the output sankey diagram
        config: a dictionary of the input config yaml contents
        
    Returns:
        a pdf of the sankey diagram for the run
    """
    fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = sankey_dict["label"],
    ),
    link = dict(
      source = sankey_dict["source"],
      target = sankey_dict["target"],
      value = sankey_dict["value"],
  ))])
    fig.write_image("sankey.png")
    
    