import os
from langchain.vectorstores import Chroma
from lib.const import META_DIR

VECTORDB_DIR = META_DIR + 'vectordb'
COLLECTION_METADATA = 'collection_metadata'
DEFAULT_COLLECTION_NAME = '__default'
BUF_SIZE = 4096

class VectorStore:
    def __init__(self, llm):
        self.llm = llm
        os.makedirs(VECTORDB_DIR, exist_ok=True)

    def __get_vectorstore(self, ds_type, collection):
        persist_dir = os.path.join(VECTORDB_DIR, ds_type)
        return Chroma(collection_name=collection,
                      embedding_function=self.llm.embeddings,
                      persist_directory=persist_dir)

    def update(self, ds_type, data):
        self.__get_vectorstore(ds_type, data.Collection) \
                .add_texts([data.Document], [data.Metadata], [data.Id])

    def similarity_search(self, ds_type, collection, query):
        return self.__get_vectorstore(ds_type, collection).similarity_search(query)
