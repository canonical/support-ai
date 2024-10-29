"""
This module provides a factory for creating Hugging Face-based
language models and embeddings.
"""
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from support_ai.lib import const
from support_ai.lib.model_manager.model_factory import ModelFactory


class HuggingFaceFactory(ModelFactory):
    """
    Factory class for creating Hugging Face-based language models
    and embeddings.
    """

    def __init__(self, llm_config):
        """
        Initializes the HuggingFaceFactory with model configuration.

        Args:
            llm_config: Configuration dictionary containing the model name
                        or path.

        Raises:
            ValueError: If the model name or path is not specified in
                        the config.
        """
        self.model = llm_config[const.CONFIG_MODEL]
        if not self.model:
            raise ValueError(f'Missing {const.CONFIG_MODEL} in llm config')

    def create_llm(self):
        """
        Creates an instance of the Hugging Face language model.

        Returns:
            HuggingFacePipeline: A Hugging Face language model instance.
        """
        return HuggingFacePipeline.from_model_id(model_id=self.model,
                                                 task='text-generation')

    def create_embeddings(self):
        """
        Creates an instance of Hugging Face embeddings.

        Returns:
            HuggingFaceEmbeddings: A Hugging Face embeddings instance.
        """
        return HuggingFaceEmbeddings(model_name=self.model)
