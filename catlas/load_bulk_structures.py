import os.path
import json
from monty.json import MontyDecoder

import catlas


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
    """
    Load bulks from an ase.db

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

    with open(path, "rb") as f:
        bulk_list = json.load(f, cls=MontyDecoder)

    return bulk_list
