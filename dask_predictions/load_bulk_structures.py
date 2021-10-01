from ase.db import connect
import os.path

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_ocdata_bulks():
    with connect(dir_path+'/bulk_structures/ocdata_bulks.db') as db:

        # Turn each entry into a dictionary that will become the dataframe columns
        bulk_list = []
        for row in db.select():
            bulk_list.append({'bulk.atoms': row.toatoms(),
                              'bulk.mpid': row.mpid,
                              'bulk.data_source': 'ocdata_bulks',
                              'bulk.natoms': row.natoms,
                              'bulk.xc': 'RPBE',
                              'bulk.symbols': row.symbols,
                              'bulk.nsymbols': len(row.symbols)})

        return bulk_list
