"""
Vector Store for Embedding and Similarity Search
"""

import os

from langchain_community.vectorstores import Chroma

from . import const

VECTORDB_DIR = const.META_DIR + 'vectordb'
COLLECTION_METADATA = 'collection_metadata'
DEFAULT_COLLECTION_NAME = '__default'


class VectorStore:
    """
    A class to manage vector storage and similarity search using Chroma.

    Attributes:
        VECTORDB_DIR (str): Directory for vector database persistence.
    """

    def __init__(self):
        """
        Initializes the VectorStore, ensuring the storage directory exists.
        """
        os.makedirs(VECTORDB_DIR, exist_ok=True)

    def __get_vectorstore(self, ds_type, embedding):
        """
        Initializes a Chroma vector store for a specified data source type.

        Args:
            ds_type: Type identifier for the data source.
            embedding: Embedding function to convert text into vector format.

        Returns:
            Chroma: A Chroma vector store instance configured for the specified
                    data source.
        """
        persist_dir = os.path.join(VECTORDB_DIR, ds_type)
        return Chroma(embedding_function=embedding,
                      persist_directory=persist_dir)

    def update(self, ds_type, embedding, data):
        """
        Adds a new document to the vector store.

        Args:
            ds_type: Type identifier for the data source.
            embedding: Embedding function to convert text into vector format.
            data: Document data with attributes `document`, `metadata`,
                  and `id`.
        """
        self.__get_vectorstore(ds_type, embedding) \
            .add_texts([data.document], [data.metadata], [data.id])

    def similarity_search(self, ds_type, embedding, query):
        """
        Performs a similarity search on the vector store based on the query.

        Args:
            ds_type: Type identifier for the data source.
            embedding: Embedding function for text-to-vector
                                  conversion.
            query: The search query text.

        Returns:
            list: List of search results ranked by similarity to the query.
        """
        return self.__get_vectorstore(ds_type,
                                      embedding).similarity_search(query)
