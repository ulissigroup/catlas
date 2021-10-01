from ocdata.bulk_obj import Bulk


class CustomBulk(Bulk):
    def __init__(self, bulk_atoms):
        self.bulk_atoms = bulk_atoms
