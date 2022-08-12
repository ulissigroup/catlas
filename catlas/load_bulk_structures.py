"""Function to load bulks from an ase.db."""
import os.path

import numpy as np
import pandas as pd
import pickle

import catlas
import catlas.cache_utils


required_fields = (
    "atoms",
    "mpid",
    "natoms",
    "xc",
    "nelements",
    "elements",
)  # These fields are expected to exist in every input file that doesn't allow them to be directly calculated


def load_bulks(bulk_path):
    """
    Load bulks from an ase.db

    Args:
        bulk_path: a relative path (from the main catlas directory) to the ase.db
    """

    path = "%s/%s" % (
        os.path.join(os.path.dirname(catlas.__file__), os.pardir),
        bulk_path,
    )

    path_name, _ = os.path.splitext(path)
    db_name = path_name.split("/")[-1]

    with open(path, "rb") as f:
        bulk_list = pickle.load(f)

    return bulk_list
