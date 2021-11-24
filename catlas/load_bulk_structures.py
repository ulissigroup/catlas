from ase.db import connect
import os.path
import numpy as np
import pandas as pd
from .dask_utils import SizeDict
import catlas

required_fields = (
    "atoms",
    "mpid",
    "natoms",
    "xc",
    "nelements",
    "elements",
)  # These fields are expected to exist in every input file that doesn't allow them to be directly calculated


def load_bulks(bulk_path):

    path = "%s/%s" % (
        os.path.join(os.path.dirname(catlas.__file__), os.pardir),
        bulk_path,
    )

    path_name, ext = os.path.splitext(path)
    source_name = path_name.split("/")[-1]

    if ext == ".db":
        return load_bulks_from_db(path, source_name)
    elif (ext == ".pkl") or (ext == ".json"):
        return load_bulks_from_df(path, source_name)


def load_bulks_from_db(db_path, db_name):
    with connect(db_path) as db:

        # Turn each entry into a dictionary that will become the dataframe columns
        bulk_list = []
        for row in db.select():
            bulk_list.append(
                SizeDict(
                    {
                        "bulk_atoms": row.toatoms(),
                        "bulk_mpid": row.mpid,
                        "bulk_data_source": db_name,
                        "bulk_natoms": row.natoms,
                        "bulk_xc": "RPBE",
                        "bulk_nelements": len(
                            np.unique(row.toatoms().get_chemical_symbols())
                        ),
                        "bulk_elements": np.unique(
                            row.toatoms().get_chemical_symbols()
                        ),
                    }
                )
            )

        return bulk_list


def load_bulks_from_df(df_path, df_name):

    _, ext = os.path.splitext(df_path)
    if ext == ".pkl":
        bulk_df = pd.DataFrame(pd.read_pickle(df_path))
    elif ext == ".json":
        bulk_df = pd.DataFrame(pd.read_json(df_path))

    bulk_df = bulk_df.loc[:, required_fields]
    bulk_df["source"] = df_name
    bulk_df = bulk_df.rename(lambda x: "bulk_" + x, axis=1)
    bulk_list = bulk_df.to_dict("records")
    return bulk_list
