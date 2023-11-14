from lib.datasources.ds_querier import DSQuerier

class Chain:
    def __init__(self, datasources):
        self.ds_querier = DSQuerier(datasources)

    def ask(self, query, ds_type=None):
        return self.ds_querier.query(query, ds_type)
