"""
Modified from Open-Catalyst-Project/Open-Catalyst-Dataset
This submodule contains the scripts that the we used to sample the adsorption
structures.
Note that some of these scripts were taken from
[GASpy](https://github.com/ulissigroup/GASpy) with permission of author.
"""

__authors__ = ["Kevin Tran", "Aini Palizhati", "Siddharth Goyal", "Zachary Ulissi"]
__email__ = ["ktran@andrew.cmu.edu"]

import numpy as np
import pickle
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.ext.matproj import MPRester
from pymatgen.core.surface import (
    SlabGenerator,
    get_symmetrically_distinct_miller_indices,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def enumerate_surfaces_for_saving(bulk_structure, max_miller):
    """
    Enumerate all the symmetrically distinct surfaces of a bulk structure. It
    will not enumerate surfaces with Miller indices above the `max_miller`
    argument. Note that we also look at the bottoms of surfaces if they are
    distinct from the top. If they are distinct, we flip the surface so the bottom
    is pointing upwards.

    Args:
        bulk_atoms (ase.Atoms): object of the bulk you want to enumerate surfaces from.
        max_miller (int) value indicating the maximum Miller index of the surfaces
            to be enumerated.

    Returns:
        list[tuple]: pymatgen.structure.Structure objects for surfaces we have enumerated,
            the Miller indices, floats for the shifts, and Booleans for "top".
    """

    all_slabs_info = []
    for millers in get_symmetrically_distinct_miller_indices(
        bulk_structure, max_miller
    ):
        slab_gen = SlabGenerator(
            initial_structure=bulk_structure,
            miller_index=millers,
            min_slab_size=7.0,
            min_vacuum_size=20.0,
            lll_reduce=False,
            center_slab=True,
            primitive=True,
            max_normal_search=1,
        )
        slabs = slab_gen.get_slabs(
            tol=0.3, bonds=None, max_broken_bonds=0, symmetrize=False
        )

        # If the bottoms of the slabs are different than the tops, then we want
        # to consider them, too
        flipped_slabs_info = [
            (flip_struct(slab), millers, slab.shift, False)
            for slab in slabs
            if is_structure_invertible(slab) is False
        ]

        # Concatenate all the results together
        slabs_info = [(slab, millers, slab.shift, True) for slab in slabs]
        all_slabs_info.extend(slabs_info + flipped_slabs_info)
    return all_slabs_info


def is_structure_invertible(structure):
    """
    This function figures out whether or not an `pymatgen.Structure` object has
    symmetricity. In this function, the affine matrix is a rotation matrix that
    is multiplied with the XYZ positions of the crystal. If the z,z component
    of that is negative, it means symmetry operation exist, it could be a
    mirror operation, or one that involves multiple rotations/etc. Regardless,
    it means that the top becomes the bottom and vice-versa, and the structure
    is the symmetric. i.e. structure_XYZ = structure_XYZ*M.
    In short:  If this function returns `False`, then the input structure can
    be flipped in the z-direction to create a new structure.

    Args:
        structure (pymatgen.structure.Structure): surface pmg object.

    Returns:
        bool: indicating whether or not the surface object is
            symmetric in z-direction (i.e. symmetric with respect to x-y plane).
    """
    # If any of the operations involve a transformation in the z-direction,
    # then the structure is invertible.
    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    for operation in sga.get_symmetry_operations():
        xform_matrix = operation.affine_matrix
        z_xform = xform_matrix[2, 2]
        if z_xform == -1:
            return True
    return False


def flip_struct(struct):
    """
    Flips an atoms object upside down. Normally used to flip surfaces.
    Arg:
        struct (pymatgen.structure.Structure): surface object
    Returns:
        (pymatgen.structure.Structure): The same structure object that was fed as an
            argument, but flipped upside down.
    """
    atoms = AseAtomsAdaptor.get_atoms(struct)
    # This is black magic wizardry to me. Good look figuring it out.
    atoms.wrap()
    atoms.rotate(180, "x", rotate_cell=True, center="COM")
    if atoms.cell[2][2] < 0.0:
        atoms.cell[2] = -atoms.cell[2]
    if np.cross(atoms.cell[0], atoms.cell[1])[2] < 0.0:
        atoms.cell[1] = -atoms.cell[1]
    atoms.wrap()

    flipped_struct = AseAtomsAdaptor.get_structure(atoms)

    # Add the wyckoff positions back
    wyckoffs = [site.full_wyckoff for site in struct]
    flipped_struct.add_site_property("full_wyckoff", wyckoffs)
    return flipped_struct

def enumerate_custom_surfaces_for_saving(bulk_structure, config):
    with open(config["input_options"]["custom_slab_file"], 'rb') as f:
        custom_slab_file = pickle.load(f)
    custom_slab_file = pd.DataFrame(custom_slab_file)
    
    slabs=[]
    for r in range(0,len(custom_slab_file)):
        with MPRester("MGOdX3P4nI18eKvE") as mpr:
            test_structure = mpr.get_structure_by_material_id(custom_slab_file.bulk_id[r],
                                                            final=True, 
                                                            conventional_unit_cell
                                                            =True)
        if test_structure == bulk_structure:
            slab_gen = SlabGenerator(
            initial_structure=bulk_structure,
            miller_index=custom_slab_file.miller_index[r],
            min_slab_size=7.0,
            min_vacuum_size=20.0,
            lll_reduce=False,
            center_slab=True,
            primitive=True,
            max_normal_search=1,
        )
        slab = slab_gen.get_slab(shift = custom_slab_file.shift[r],
            tol=0.3, energy=None,
        )
        slabs.append(slab)
    all_slabs_info = [(slab, slab.miller_index, slab.shift, True) for slab in slabs]
    return all_slabs_info