"""General use utils to support catlas."""

import numpy as np


def get_center_of_mass(pmg_struct):
    """
    Calculates the center of mass of a pmg structure.

    Args:
        pmg_struct (pymatgen.core.structure.Structure): pymatgen structure to be
            considered.

    Returns:
        numpy.ndarray: the center of mass
    """
    weights = [s.species.weight for s in pmg_struct]
    center_of_mass = np.average(pmg_struct.frac_coords, weights=weights, axis=0)
    return center_of_mass

def surface_area(slab):
    """
    Gets cross section surface area of the slab.
    Args:
        slab (pymatgen.structure.Structure): PMG Structure representation of a slab.

    Returns:
        (float): surface area

    """
    m = slab.lattice.matrix
    return np.linalg.norm(np.cross(m[0], m[1]))