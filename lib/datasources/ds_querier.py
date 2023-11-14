from lib.const import CONFIG_SF, CONFIG_KB
from lib.vectorstore import VectorStore


class DSQuerier:
    def __init__(self, datasources):
        self.vector_store = VectorStore()
        self.datasources = datasources

    def __judge_ds_type(self, query):
        return CONFIG_SF

    def __get_ds(self, ds_type):
        if ds_type not in self.datasources:
            raise ValueError(f'Unknown datasource type: {ds_type}')
        return self.datasources[ds_type]

    def query(self, query, ds_type=None):
        if ds_type is None:
            ds_type = self.__judge_ds_type(query)
        ds = self.__get_ds(ds_type)
        docs = self.vector_store.similarity_search(ds_type,
                                                  ds.model_manager.embeddings,
                                                  query)
        return ds.get_content(docs[0])
