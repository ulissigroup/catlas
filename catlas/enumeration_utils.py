"""
Modified from Open-Catalyst-Project/Open-Catalyst-Dataset
This submodule contains the scripts that the we used to sample the adsorption
structures.
Note that some of these scripts were taken from
[GASpy](https://github.com/ulissigroup/GASpy) with permission of author.
"""

import numpy as np
import math
import os
import pickle

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import (
    SlabGenerator,
    get_symmetrically_distinct_miller_indices,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ase.constraints import FixAtoms
from pymatgen.analysis.local_env import VoronoiNN

MIN_XY = 8.0

from catlas.general_utils import get_center_of_mass


def enumerate_surfaces_for_saving(bulk_structure, max_miller):
    """
    Enumerate all the symmetrically distinct surfaces of a bulk structure. It
    will not enumerate surfaces with Miller indices above the `max_miller`
    argument. Note that we also look at the bottoms of surfaces if they are
    distinct from the top. If they are distinct, we flip the surface so the bottom
    is pointing upwards.

    Args:
        bulk_structure (pymatgen.structure.Structure): object of the bulk you want
            to enumerate surfaces from.
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
        (bool): indicating whether or not the surface object is
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

    Args:
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


def constrain_surface(atoms):
    """
    This function fixes sub-surface atoms of a surface. Also works on systems
    that have surface + adsorbate(s), as long as the bulk atoms are tagged with
    `0`, surface atoms are tagged with `1`, and the adsorbate atoms are tagged
    with `2` or above. This function is used for both surface atoms and the combined
    surface+adsorbate

    Args:
        atoms (ase.Atoms): class of the surface system. The tags of these atoms must
            be set such that any bulk/surface atoms are tagged with `0` or `1`,
            resectively, and any adsorbate atom is tagged with a 2 or above.

    Returns:
        (ase.Atoms): A copy of the `atoms` argument, but where the appropriate
            atoms are constrained.
    """
    # Work on a copy so that we don't modify the original
    atoms = atoms.copy()

    # We'll be making a `mask` list to feed to the `FixAtoms` class. This list
    # should contain a `True` if we want an atom to be constrained, and `False`
    # otherwise
    mask = [True if atom.tag == 0 else False for atom in atoms]
    atoms.constraints += [FixAtoms(mask=mask)]
    return atoms


class Surface:
    """
    This class handles all things with a surface.
    Create one with a bulk and one of its selected surfaces

    Attributes
    ----------
    bulk_struct : pymatgen.structure.Structure
        An object of the bulk structure.
    surface_struct : pymatgen.structure.Structure
        An object of the surface structure.
    surface_atoms : ase.Atoms
        atoms of the surface
    constrained_surface : ase.Atoms
        constrained version of surface_atoms
    millers : tuple
        miller indices of the surface
    shift : float
        shift applied in the c-direction of bulk unit cell to get a termination
    top : boolean
        indicates the top or bottom termination of the pymatgen generated slab
    """

    def __init__(self, bulk_struct, surface_info):
        """
        Initialize the surface object, tag atoms, and constrain the surface.

        Args:
            bulk_struct (pymatgen.structure.Structure):  An object of the bulk structure.
            surface_info: tuple containing atoms, millers, shift, top
        """
        self.bulk_struct = bulk_struct
        surface_struct, self.millers, self.shift, self.top = surface_info

        self.surface_struct = self.tile_structure(surface_struct)
        self.surface_atoms = AseAtomsAdaptor.get_atoms(self.surface_struct)

        self.tag_surface_atoms(self.bulk_struct, self.surface_struct)
        self.constrained_surface = constrain_surface(self.surface_atoms)

    def tile_structure(self, structure):
        """
        This function will repeat an atoms structure in the x and y direction until
        the x and y dimensions are at least as wide as the MIN_XY constant.

        Args:
            structure (pymatgen.structure.Structure):  An object of the structure to tile.

        Returns:
             (pymatgen.structure.Structure):  An object of the structure post-tile.
        """
        x_length = structure.lattice.abc[0]
        y_length = structure.lattice.abc[1]
        nx = int(math.ceil(MIN_XY / x_length))
        ny = int(math.ceil(MIN_XY / y_length))
        structure.make_supercell([[nx, 0, 0], [0, ny, 0], [0, 0, 1]])
        return structure

    def tag_surface_atoms(self, bulk_struct, surface_struct):
        """
        Sets the tags of an `ase.Atoms` object. Any atom that we consider a "bulk"
        atom will have a tag of 0, and any atom that we consider a "surface" atom
        will have a tag of 1.

        Args:
            bulk_struct (pymatgen.structure.Structure):  An object of the bulk structure.
            surface_struct (pymatgen.structure.Structure):  An object of the surface structure.
        """
        voronoi_tags = self._find_surface_atoms_with_voronoi(
            bulk_struct, surface_struct
        )
        self.surface_atoms.set_tags(voronoi_tags)

    def _find_surface_atoms_with_voronoi(self, bulk_struct, surface_struct):
        """
        Labels atoms as surface or bulk atoms according to their coordination
        relative to their bulk structure. If an atom's coordination is less than it
        normally is in a bulk, then we consider it a surface atom. We calculate the
        coordination using pymatgen's Voronoi algorithms.

        Args:
            bulk_struct (pymatgen.structure.Structure):  An object of the bulk structure.
            surface_struct (pymatgen.structure.Structure):  An object of the surface structure.

        Returns:
            (list): A list of 0's and 1's whose indices align with the atoms in
                surface_struct. 0's indicate a subsurface atom and 1 indicates a surface atom.
        """
        # Initializations
        center_of_mass = get_center_of_mass(surface_struct)
        bulk_cn_dict = self.calculate_coordination_of_bulk_struct(bulk_struct)
        voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection

        tags = []
        for idx, site in enumerate(surface_struct):

            # Tag as surface atom only if it's above the center of mass
            if site.frac_coords[2] > center_of_mass[2]:
                try:

                    # Tag as surface if atom is under-coordinated
                    cn = voronoi_nn.get_cn(surface_struct, idx, use_weights=True)
                    cn = round(cn, 5)
                    if cn < bulk_cn_dict[site.full_wyckoff]:
                        tags.append(1)
                    else:
                        tags.append(0)

                # Tag as surface if we get a pathological error
                except RuntimeError:
                    tags.append(1)

            # Tag as bulk otherwise
            else:
                tags.append(0)
        return tags

    def calculate_coordination_of_bulk_struct(self, bulk_struct):
        """
        Finds all unique sites in a bulk structure and then determines their
        coordination number. Then parses these coordination numbers into a
        dictionary whose keys are the elements of the atoms and whose values are
        their possible coordination numbers.
        For example: `bulk_cns = {'Pt': {3., 12.}, 'Pd': {12.}}`

        Args:
            bulk_struct (pymatgen.structure.Structure):  An object of the bulk structure.

        Returns:
            (dict): A dict whose keys are the wyckoff values in the bulk_struct
                and whose values are the coordination numbers of that site.
        """
        voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection

        # Object type conversion so we can use Voronoi
        sga = SpacegroupAnalyzer(bulk_struct)

        # We'll only loop over the symmetrically distinct sites for speed's sake
        bulk_cn_dict = {}
        for idx, site in enumerate(bulk_struct):
            if site.full_wyckoff not in bulk_cn_dict:
                cn = voronoi_nn.get_cn(bulk_struct, idx, use_weights=True)
                cn = round(cn, 5)
                bulk_cn_dict[site.full_wyckoff] = cn
        return bulk_cn_dict
