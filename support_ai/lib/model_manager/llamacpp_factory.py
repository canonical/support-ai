
"""
This module provides a factory for creating LlamaCpp-based language models
and embeddings.
"""
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from support_ai.lib import const
from support_ai.lib.model_manager.model_factory import ModelFactory


class LlamaCppFactory(ModelFactory):
    """
    Factory class for creating LlamaCpp-based language models and embeddings.
    """

    def __init__(self, llm_config):
        """
        Initializes the LlamaCppFactory with model configuration.

        Args:
            llm_config: Configuration dictionary containing the model path.

        Raises:
            ValueError: If the model path is not specified in llm_config.
        """

        self.model = llm_config[const.CONFIG_MODEL]
        if not self.model:
            raise ValueError(f'Missing {const.CONFIG_MODEL} in llm config')

    def create_llm(self):
        """
        Creates an instance of the LlamaCpp language model.

        Returns:
            LlamaCpp: A LlamaCpp model instance with a specified context size.
        """
        return LlamaCpp(model_path=self.model, n_ctx=4096)

    def create_embeddings(self):
        """
        Creates an instance of LlamaCpp embeddings.

        Returns:
            LlamaCppEmbeddings: A LlamaCpp embeddings instance with a
                                specified context size.
        """
        return LlamaCppEmbeddings(model_path=self.model, n_ctx=4096)
