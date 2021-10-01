from ase.db import connect
import os.path
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


def load_ocdata_bulks():
    with connect(dir_path + "/bulk_structures/ocdata_bulks.db") as db:

        # Turn each entry into a dictionary that will become the dataframe columns
        bulk_list = []
        for row in db.select():
            bulk_list.append(
                {
                    "bulk_atoms": row.toatoms(),
                    "bulk_mpid": row.mpid,
                    "bulk_data_source": "ocdata_bulks",
                    "bulk_natoms": row.natoms,
                    "bulk_xc": "RPBE",
                    "bulk_symbols": row.symbols,
                    "bulk_nsymbols": len(row.symbols),
                    "bulk_elements": np.unique(row.toatoms().get_chemical_symbols()),
                }
            )

        return bulk_list
