"""
Functions to assess nuclearity on catlas entries.
Original code taken from https://github.com/ulissigroup/NuclearityCalculation
Original author: Unnatti Sharma
"""

from ase import neighborlist
from ase.neighborlist import natural_cutoffs
import graph_tool as gt
from graph_tool import topology
import numpy as np


def get_nuclearity(entry):
    """
    Function to get the nuclearity for each element in a surface
    Args:
        entry: a catlas-like entry object

    Returns:
        entry: a catlas-like entry object with nuclearity info added
    """
    elements = entry["bulk_elements"]
    slab_atoms = entry["slab_surface_object"].surface_atoms
    replicated_slab_atoms = slab_atoms.repeat((2, 2, 1))

    # Grab connectivity matricies
    overall_connectivity_matrix = get_connectivity_matrix(slab_atoms)
    overall_connectivity_matrix_rep = get_connectivity_matrix(replicated_slab_atoms)

    # Grab surface atom idxs
    surface_indices = [idx for idx, tag in enumerate(slab_atoms.get_tags()) if tag == 1]
    surface_indices_rep = [
        idx for idx, tag in enumerate(replicated_slab_atoms.get_tags()) if tag == 1
    ]

    # Iterate over atoms and assess nuclearity
    output_dict = {}
    for element in elements:
        surface_atoms_of_element = [
            atom.symbol == element and atom.index in surface_indices
            for atom in slab_atoms
        ]
        surface_atoms_of_element_rep = [
            atom.symbol == element and atom.index in surface_indices_rep
            for atom in replicated_slab_atoms
        ]

        if sum(surface_atoms_of_element) == 0:
            output_dict[element] = {"nuclearity": 0, "nuclearities": []}

        else:
            hist = get_nuclearity_neighbor_counts(
                surface_atoms_of_element, overall_connectivity_matrix
            )
            hist_rep = get_nuclearity_neighbor_counts(
                surface_atoms_of_element_rep, overall_connectivity_matrix_rep
            )
            output_dict[element] = evaluate_infiniteness(hist, hist_rep)
    entry["nuclearity_info"] = output_dict
    return entry


def get_nuclearity_neighbor_counts(surface_atoms_of_element, connectivity_matrix):
    """
    Function that counts the like surface neighbors for surface atoms
    Args:
        surface_atoms_of_element: list of all surface atoms which are of a specific element
        connectivity_matrix: matrix memorializing which atoms in the slab are connected

    Returns:
        hist: counts of neighbor groups
    """
    connectivity_matrix = connectivity_matrix[surface_atoms_of_element, :]
    connectivity_matrix = connectivity_matrix[:, surface_atoms_of_element]
    graph = gt.Graph(directed=False)
    graph.add_vertex(n=connectivity_matrix.shape[0])
    graph.add_edge_list(np.transpose(connectivity_matrix.nonzero()))
    labels, hist = topology.label_components(graph, directed=False)
    return hist


def evaluate_infiniteness(hist, hist_rep):
    """
    Function that compares the connected counts between the minimal slab
    and a repeated slab to classify the type of infiniteness.
    Args:
        hist: list of nuclearities observed in minimal slab
        hist_rep: list of nuclearities observed in replicated slab

    Returns:
        nuclearity dict: the max nuclearity and all nuclearities for the
        element on that surface
    """
    if max(hist) == max(hist_rep):
        return {"nuclearity": max(hist), "nuclearities": hist}
    elif max(hist) == 0.5 * max(hist_rep):
        return {"nuclearity": "semi-finite", "nuclearities": hist}
    elif max(hist) == 0.25 * max(hist_rep):
        return {"nuclearity": "infinite", "nuclearities": hist}
    else:
        return {"nuclearity": "somewhat-infinite", "nuclearities": hist}


def get_connectivity_matrix(slab_atoms):
    """
    Function that gets a connectivity matrix by looking at nearest neighbors.
    Args:
        slab_atoms: atoms object of the slab

    Returns:
        overall_connectivity_matrix
    """
    # For initial atoms
    surface_indices = [idx for idx, tag in enumerate(slab_atoms.get_tags()) if tag == 1]
    cutOff = natural_cutoffs(slab_atoms)
    neighborList = neighborlist.NeighborList(
        cutOff, self_interaction=False, bothways=True
    )
    neighborList.update(slab_atoms)
    overall_connectivity_matrix = neighborList.get_connectivity_matrix()
    return overall_connectivity_matrix
