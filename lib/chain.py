from lib.datasources.ds_querier import DSQuerier


class Chain:
    def __init__(self, config):
        self.ds_querier = DSQuerier(config)

    def ask(self, query, ds_type=None):
        return self.ds_querier.query(query, ds_type)
