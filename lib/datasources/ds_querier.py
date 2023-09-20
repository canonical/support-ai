from lib.const import CONFIG_SF

class DSQuerier:
    def __init__(self, vector_store, datasources):
        self.vector_store = vector_store
        self.datasources = datasources

    def __judge_ds_type(self, query):
        return CONFIG_SF

    def __judge_collection(self, query):
        return 'ceph'

    def __get_ds(self, ds_type):
        if ds_type not in self.datasources:
            raise ValueError(f'Unknown datasource: {ds_type}')
        return self.datasources[ds_type]

    def query(self, query):
        ds_type = self.__judge_ds_type(query)
        collection = self.__judge_collection(query)
        ds = self.__get_ds(ds_type)
        docs = self.vector_store.similarity_search(ds_type, collection, query)

        for doc in docs:
            print(ds.get_content(doc))
