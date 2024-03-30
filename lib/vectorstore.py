import os
from langchain.vectorstores import Chroma
from lib.const import META_DIR

VECTORDB_DIR = META_DIR + 'vectordb'
COLLECTION_METADATA = 'collection_metadata'
DEFAULT_COLLECTION_NAME = '__default'


class VectorStore:
    def __init__(self):
        os.makedirs(VECTORDB_DIR, exist_ok=True)

    def __get_vectorstore(self, ds_type, embedding):
        persist_dir = os.path.join(VECTORDB_DIR, ds_type)
        return Chroma(embedding_function=embedding,
                      persist_directory=persist_dir)

    def update(self, ds_type, embedding, data):
        self.__get_vectorstore(ds_type, embedding) \
                .add_texts([data.document], [data.metadata], [data.id])

    def similarity_search(self, ds_type, embedding, query):
        return self.__get_vectorstore(ds_type, embedding).similarity_search(query)
