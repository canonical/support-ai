"""
This module provides the DSQuerier class, which is responsible for
querying different data sources based on the provided configuration
and user queries.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .. import const
from ..context import BaseContext
from ..vectorstore import VectorStore
from .utils import get_datasources


CLASSIFICATION_PROMPT = """
Classify the question into salesforce or knowledgebase.

Example:
    Question: Some issues happened, is there similar discussions?
    Answer: salesforce
    Question: Give me operational steps to resolve certain issue
    Answer: knowledgebase

Do not respond with the answer other than salesforce, knowledgebase.

Question: {query}
Answer:
""".strip()


class DSQuerier(BaseContext):
    """
    A class for querying data sources based on user queries.
    """

    def __init__(self, config):
        super().__init__(config)
        if const.CONFIG_BASIC_MODEL not in config:
            raise ValueError(
                f'The config doesn\'t contain {const.CONFIG_BASIC_MODEL}')

        self.model = self.model_manager.get_model(
                        config[const.CONFIG_BASIC_MODEL])
        self.datasources = get_datasources(config)
        self.vector_store = VectorStore()

    def __judge_ds_type(self, query):
        """
        Determines the type of data source based on the query.

        Args:
            query: The user query to classify.

        Returns:
            str: The type of the data source.
        """
        if len(self.datasources) == 1:
            return list(self.datasources.keys())[0]
        prompt = PromptTemplate.from_template(CLASSIFICATION_PROMPT)
        chain = (
                {'query': RunnablePassthrough()}
                | prompt
                | self.model.llm
                | StrOutputParser()
                )
        ds_type = chain.invoke(query)
        if ds_type not in self.datasources:
            return list(self.datasources.keys())[0]
        return ds_type

    def get_ds(self, ds_type):
        """
        Retrieves the specified data source.

        Args:
            ds_type: The type of data source to retrieve.

        Returns:
            Data source instance corresponding to the ds_type.

        Raises:
            ValueError: If the specified data source type is unknown.
        """
        if ds_type not in self.datasources:
            raise ValueError(f'Unknown datasource type: {ds_type}')
        return self.datasources[ds_type]

    def query(self, query, ds_type=None):
        """
        Executes a query against the appropriate data source.

        Args:
            query: The user query to execute.
            ds_type: The type of data source. If None,
                     it will be determined.

        Returns:
            Tuple: A tuple containing the data source instance and the
                   retrieved documents.
        """
        if ds_type is None:
            ds_type = self.__judge_ds_type(query)

        ds = self.get_ds(ds_type)
        docs = self.vector_store.similarity_search(ds_type,
                                                   ds.model_manager.embeddings,
                                                   query)
        return ds, docs[0]
