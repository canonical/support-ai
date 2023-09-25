from lib.const import CONFIG_SF

SCORE_THRESHOLD = 100

class DSQuerier:
    def __init__(self, vector_store, datasources):
        self.vector_store = vector_store
        self.datasources = datasources

    def __judge_ds_type(self, query):
        return CONFIG_SF

    def __get_ds(self, ds_type):
        if ds_type not in self.datasources:
            raise ValueError(f'Unknown datasource: {ds_type}')
        return self.datasources[ds_type]

    def query(self, query):
        ds_type = self.__judge_ds_type(query)
        ds = self.__get_ds(ds_type)
        collections = self.vector_store.list_collections(ds_type)
        docs_with_score = []

        for collection in collections:
            docs_with_score += self.vector_store.similarity_search(ds_type, collection, query)

        docs_with_score.sort(key=lambda val: val[1])
        for doc, score in docs_with_score:
            if score >= SCORE_THRESHOLD:
                break
            yield (ds.get_summary_prompt(), ds.get_content(doc))
