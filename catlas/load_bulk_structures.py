"""Function to load bulks from an ase.db."""
import os.path

import numpy as np
from ase.db import connect

import catlas
import catlas.cache_utils


required_fields = (
    "atoms",
    "mpid",
    "natoms",
    "xc",
    "nelements",
    "elements",
)  # These fields are expected to exist in every input file that doesn't allow them to
# be directly calculated


def load_bulks(bulk_path):
    """Load bulks from an ase.db

    Args:
        bulk_path (str): a relative path (from the main catlas directory) to the ase.db

    Returns:
        list[dict]: A list of dictionaries corresponding to bulk structures.
    """
    path = "%s/%s" % (
        os.path.join(os.path.dirname(catlas.__file__), os.pardir),
        bulk_path,
    )

    path_name, _ = os.path.splitext(path)
    db_name = path_name.split("/")[-1]

    with connect(path) as db:

        # Turn each entry into a dictionary that will become the dataframe columns
        bulk_list = []
        for row in db.select():
            bulk_list.append(
                {
                    "bulk_atoms": row.toatoms(),
                    "bulk_id": row.bulk_id,
                    "bulk_data_source": db_name,
                    "bulk_natoms": row.natoms,
                    "bulk_xc": "RPBE",
                    "bulk_nelements": len(
                        np.unique(row.toatoms().get_chemical_symbols())
                    ),
                    "bulk_elements": np.unique(row.toatoms().get_chemical_symbols()),
                    "bulk_e_above_hull": row.energy_above_hull,
                    "bulk_band_gap": row.band_gap,
                }
            )

        return bulk_list
