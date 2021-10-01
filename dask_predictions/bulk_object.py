from ocdata.bulk_obj import Bulk


class CustomBulk(Bulk):
    def __init__(self, bulk):
        bulk_atoms, mpid = bulk
        self.bulk_atoms = bulk_atoms
        self.mpid = mpid
