"""
This module provides a factory for creating Ollama-based chat models
and embeddings.
"""
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from support_ai.lib import const
from support_ai.lib.model_manager.model_factory import ModelFactory


class OllamaFactory(ModelFactory):
    """
    Factory class for creating Ollama-based chat models and embeddings.
    """

    def __init__(self, config):
        """
        Initializes the OllamaFactory with model configuration.

        Args:
            config: Configuration dictionary containing the model name or path.

        Raises:
            ValueError: If the model name or path is not specified in
                        the config.
        """
        self.model = config[const.CONFIG_MODEL]
        if not self.model:
            raise ValueError(f'Missing {const.CONFIG_MODEL} in llm config')

    def create_llm(self):
        """
        Creates an instance of the Ollama chat model.

        Returns:
            ChatOllama: An Ollama chat model instance.
        """
        return ChatOllama(model=self.model)

    def create_embeddings(self):
        """
        Creates an instance of Ollama embeddings.

        Returns:
            OllamaEmbeddings: An Ollama embeddings instance.
        """
        return OllamaEmbeddings(model=self.model)
