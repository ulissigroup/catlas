import dask.bag as db
import dask
from ase.db import connect

def load_ocdata_bulks():
    with connect('../bulk_structures/ocdata_bulks.db') as db:

        # Turn each entry into a dictionary that will become the dataframe columns
        bulk_list = []
        for row in tqdm.tqdm(db.select()):
            bulk_list.append({'bulk.atoms': row.toatoms(),
                              'bulk.mpid': row.mpid,
                              'bulk.data_source': 'ocdata_bulks',
                              'bulk.natoms': row.natoms,
                              'bulk.xc': 'RPBE',
                              'bulk.symbols': row.symbols,
                              'bulk.nsymbols': len(row.symbols)})

        return bulk_list
