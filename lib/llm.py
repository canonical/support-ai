"""Language model manager (LLM) and factory for LLMs and embeddings"""
from abc import ABC, abstractmethod
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    LlamaCppEmbeddings,
    OpenAIEmbeddings,
)
from langchain.llms import HuggingFacePipeline, LlamaCpp, OpenAI
from lib.const import CONFIG_TYPE, CONFIG_LLM_MODEL, CONFIG_HUGGINGFACE_PIPELINE, CONFIG_LLAMACPP, \
        CONFIG_OPENAI, CONFIG_LLM_OPENAI_API_KEY


class LLMFactory(ABC):
    """Abstract factory for LLMs and embeddings"""

    @abstractmethod
    def create_llm(self):
        """Create LLM from LLM config

        Returns:
            LLM: LLM object
        """
        return NotImplemented

    @abstractmethod
    def create_embeddings(self):
        """Create embeddings from LLM config

        Returns:
            Embeddings: Embeddings object
        """
        return NotImplemented


class HuggingFaceFactory(LLMFactory):
    """Factory for HuggingFace LLMs and embeddings"""

    def __init__(self, llm_config) -> None:
        self.model = llm_config[CONFIG_LLM_MODEL]
        if not self.model:
            raise ValueError("Missing model in llm config")

    def create_llm(self) -> HuggingFacePipeline:
        return HuggingFacePipeline.from_model_id(model_id=self.model, task='text-generation')

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(model_name=self.model)


class LlamaCppFactory(LLMFactory):
    """Factory for LlamaCpp LLMs and embeddings"""

    def __init__(self, llm_config) -> None:
        self.model = llm_config[CONFIG_LLM_MODEL]
        if not self.model:
            raise ValueError("Missing model in llm config")

    def create_llm(self) -> LlamaCpp:
        return LlamaCpp(model_path=self.model, n_ctx=4096)

    def create_embeddings(self) -> LlamaCppEmbeddings:
        return LlamaCppEmbeddings(model_path=self.model, n_ctx=4096)


class OpenAIFactory(LLMFactory):
    """Factory for OpenAI LLMs and embeddings"""

    def __init__(self, llm_config) -> None:
        self.model = llm_config[CONFIG_LLM_MODEL]
        self.api_key = llm_config[CONFIG_LLM_OPENAI_API_KEY]
        if not self.api_key:
            raise ValueError("Missing api_key in llm config")

    def create_llm(self) -> OpenAI:
        return OpenAI(
            openai_api_key=self.api_key,
            model_name=self.model,
        )

    def create_embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(openai_api_key=self.api_key)


_factories: dict = {
    CONFIG_HUGGINGFACE_PIPELINE: HuggingFaceFactory,
    CONFIG_LLAMACPP: LlamaCppFactory,
    CONFIG_OPENAI: OpenAIFactory
}


class LLM:
    """Language model manager (LLM)"""

    def __init__(self, llm_config) -> None:
        # get default language model (LLM) config
        if CONFIG_TYPE not in llm_config:
            raise ValueError(f'The llm configuration doesn\'t contain {CONFIG_TYPE}')
        llm_type = llm_config[CONFIG_TYPE]
        if llm_type not in _factories:
            raise ValueError(
                f"Unknown LLM type: {llm_type}. Valid types are:"
                f"llamacpp, openai, huggingface_pipeline"
            )

        # create LLM and embeddings based on LLM config
        factory: LLMFactory = _factories[llm_type](llm_config)
        self.llm: LLM = factory.create_llm()
        self.embeddings = factory.create_embeddings()
