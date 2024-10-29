"""
This module provides the OpenAIFactory class, which is a factory for creating
OpenAI language models and embeddings based on configuration settings.
"""
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from support_ai.lib import const
from support_ai.lib.model_manager.model_factory import ModelFactory


class OpenAIFactory(ModelFactory):
    """
    Factory class for creating OpenAI language model (LLM) and embeddings
    instances using configuration settings.
    """

    def __init__(self, llm_config):
        """
        Initializes the OpenAIFactory with the necessary configuration.

        Args:
            llm_config: A dictionary containing the configuration for
                        the OpenAI LLM.

        Raises:
            ValueError: If the model name or API key is missing in
                        the configuration.
        """
        self.model = llm_config[const.CONFIG_MODEL]
        if not self.model:
            raise ValueError(f'Missing {const.CONFIG_MODEL} in llm config')
        self.api_key = llm_config[const.CONFIG_LLM_OPENAI_API_KEY]
        if not self.api_key:
            raise ValueError(f'Missing {const.CONFIG_LLM_OPENAI_API_KEY} in '
                             'llm config')

    def create_llm(self):
        """
        Creates an instance of the OpenAI language model.

        Returns:
            OpenAI: An instance of the OpenAI language model.
        """
        return OpenAI(
            openai_api_key=self.api_key,
            model_name=self.model,
        )

    def create_embeddings(self) -> OpenAIEmbeddings:
        """
        Creates an instance of OpenAI embeddings.

        Returns:
            OpenAIEmbeddings: An instance of OpenAI embeddings with the
                              specified model and API key.
        """
        return OpenAIEmbeddings(model=self.model, openai_api_key=self.api_key)
