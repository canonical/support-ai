import chromadb
import os
from langchain.vectorstores import Chroma
from lib.const import META_DIR

VECTORDB_DIR = META_DIR + 'vectordb'
COLLECTION_METADATA = 'collection_metadata'
DEFAULT_COLLECTION_NAME = '__default'


class VectorStore:
    def __init__(self):
        os.makedirs(VECTORDB_DIR, exist_ok=True)

    def __get_vectorstore(self, ds_type, llm, collection):
        persist_dir = os.path.join(VECTORDB_DIR, ds_type)
        return Chroma(collection_name=collection,
                      embedding_function=llm.embeddings,
                      persist_directory=persist_dir)

    def list_collections(self, ds_type):
        persist_dir = os.path.join(VECTORDB_DIR, ds_type)
        collections = chromadb.PersistentClient(path=persist_dir).list_collections()
        return [collection.name for collection in collections]

    def update(self, ds_type, llm, data):
        self.__get_vectorstore(ds_type, llm, data.Collection) \
                .add_texts([data.Document], [data.Metadata], [data.Id])

    def similarity_search(self, ds_type, llm, collection, query):
        return self.__get_vectorstore(ds_type, llm, collection).similarity_search_with_score(query)
