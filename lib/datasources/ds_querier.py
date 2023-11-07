from lib.const import CONFIG_SF, CONFIG_KB
from lib.vectorstore import VectorStore

SCORE_THRESHOLD = 14000
SIMILAR_DOCS_NUM = 3

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

    def query(self, query):
        ds_type = self.__judge_ds_type(query)
        ds = self.__get_ds(ds_type)
        collections = self.vector_store.list_collections(ds_type)
        docs_with_score = []

        for collection in collections:
            docs_with_score += self.vector_store.similarity_search(ds_type,
                                                                   ds.model_manager.embeddings,
                                                                   collection, query)

        docs_with_score.sort(key=lambda val: val[1])
        below_score_thres_num = sum(1 if score <= SCORE_THRESHOLD else 0 
                                    for _, score in docs_with_score)
        docs_num = 1 if below_score_thres_num else SIMILAR_DOCS_NUM
        for doc, _ in docs_with_score:
            if docs_num == 0:
                break
            yield (ds_type, ds.get_summary_prompt(), ds.get_content(doc))
            docs_num -= 1
