"""Language model manager (LLM) and factory for LLMs and embeddings"""
from abc import ABC, abstractmethod
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    LlamaCppEmbeddings,
    OpenAIEmbeddings,
)
from langchain.llms import HuggingFacePipeline, LlamaCpp, OpenAI
from lib.const import CONFIG_HUGGINGFACE_PIPELINE, CONFIG_LLAMACPP, CONFIG_LLM, \
        CONFIG_MODEL_NAME, CONFIG_MODEL_PATH, CONFIG_OPENAI, CONFIG_OPENAI_API_KEY, \
        CONFIG_SETTING


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
        self.model_name: str = llm_config[CONFIG_MODEL_NAME]
        if not self.model_name:
            raise ValueError("Missing model_name in config")

    def create_llm(self) -> HuggingFacePipeline:
        return HuggingFacePipeline.from_model_id(model_id=self.model_name, task='text-generation')

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(model_name=self.model_name)


class LlamaCppFactory(LLMFactory):
    """Factory for LlamaCpp LLMs and embeddings"""

    def __init__(self, llm_config) -> None:
        self.model_path: str = llm_config[CONFIG_MODEL_PATH]
        if not self.model_path:
            raise ValueError("Missing model_path in config")

    def create_llm(self) -> LlamaCpp:
        return LlamaCpp(model_path=self.model_path, n_ctx=4096)

    def create_embeddings(self) -> LlamaCppEmbeddings:
        return LlamaCppEmbeddings(model_path=self.model_path, n_ctx=4096)


class OpenAIFactory(LLMFactory):
    """Factory for OpenAI LLMs and embeddings"""

    def __init__(self, llm_config) -> None:
        self.model_name: str = llm_config[CONFIG_MODEL_NAME]
        self.openai_api_key: str = llm_config[CONFIG_OPENAI_API_KEY]
        if not self.openai_api_key:
            raise ValueError("Missing openai_api_key in config")

    def create_llm(self) -> OpenAI:
        return OpenAI(
            openai_api_key=self.openai_api_key,
            model_name=self.model_name,
        )

    def create_embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(openai_api_key=self.openai_api_key)


_factories: dict = {
    CONFIG_HUGGINGFACE_PIPELINE: HuggingFaceFactory,
    CONFIG_LLAMACPP: LlamaCppFactory,
    CONFIG_OPENAI: OpenAIFactory
}


class LLM:
    """Language model manager (LLM)"""

    def __init__(self, config) -> None:
        # get default language model (LLM) config
        if CONFIG_LLM not in config[CONFIG_SETTING]:
            raise ValueError(f'The configuration\'s {CONFIG_SETTING} ' +
                             f'section doesn\'t contain {CONFIG_LLM}')
        llm: str = config[CONFIG_SETTING][CONFIG_LLM]
        if llm not in config:
            raise ValueError(f'The configuration doesn\'t contain {llm} section')

        if llm not in _factories:
            raise ValueError(
                f"Unknown LLM type: {llm}. Valid types are:"
                f"llamacpp, openai, huggingface_pipeline"
            )

        # create LLM and embeddings based on LLM config
        factory: LLMFactory = _factories[llm](config[llm])
        self.llm: LLM = factory.create_llm()
        self.embeddings = factory.create_embeddings()
