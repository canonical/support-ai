"""Language model manager (LLM) and factory for LLMs and embeddings"""
from abc import ABC, abstractmethod
from typing import Type
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    LlamaCppEmbeddings,
    OpenAIEmbeddings,
)
from langchain.llms import HuggingFacePipeline, LlamaCpp, OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMFactory(ABC):
    """Abstract factory for LLMs and embeddings"""

    @abstractmethod
    def create_llm(self):
        """Create LLM from LLM config

        Returns:
            LLM: LLM object
        """

    @abstractmethod
    def create_embeddings(self):
        """Create embeddings from LLM config

        Returns:
            Embeddings: Embeddings object
        """


class HuggingFaceFactory(LLMFactory):
    """Factory for HuggingFace LLMs and embeddings"""

    def __init__(self, llm_config) -> None:
        self.model_name: str = llm_config.get('model_name')
        if not self.model_name:
            raise ValueError("Missing model_name in config")

    def create_llm(self) -> HuggingFacePipeline:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=120
        )
        return HuggingFacePipeline(pipeline=pipe)

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(model_name=self.model_name)


class LlamaCppFactory(LLMFactory):
    """Factory for LlamaCpp LLMs and embeddings"""

    def __init__(self, llm_config) -> None:
        self.model_path: str = llm_config.get('model_path')
        if not self.model_path:
            raise ValueError("Missing model_path in config")

    def create_llm(self) -> LlamaCpp:
        return LlamaCpp(model_path=self.model_path, n_ctx=2048)

    def create_embeddings(self) -> LlamaCppEmbeddings:
        return LlamaCppEmbeddings(model_path=self.model_path, n_ctx=2048)


class OpenAIFactory(LLMFactory):
    """Factory for OpenAI LLMs and embeddings"""

    def __init__(self, llm_config) -> None:
        self.model_name: str = llm_config.get('model_name')
        self.openai_api_key: str = llm_config.get('openai_api_key')
        if not self.openai_api_key:
            raise ValueError("Missing openai_api_key in config")

    def create_llm(self) -> OpenAI:
        return OpenAI(
            openai_api_key=self.openai_api_key,
            model_name=self.model_name,
        )

    def create_embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(openai_api_key=self.openai_api_key)


class LLM:
    """Language model manager (LLM)"""

    def __init__(self, config: dict) -> None:
        # get default language model (LLM) config
        default_llm: str = config.get('default_llm')
        llm_config: dict = config.get(default_llm)

        factories: dict = {
            "huggingface_pipeline": HuggingFaceFactory,
            "llamacpp": LlamaCppFactory,
            "openai": OpenAIFactory
        }
        factory: Type[LLMFactory] = factories.get(default_llm)
        if not factory:
            raise ValueError(
                f"Unknown LLM type: {default_llm}. Valid types are:"
                f"llamacpp, openai, huggingface_pipeline"
            )

        # create LLM and embeddings based on LLM config
        factory: LLMFactory = factory(llm_config)
        self.llm: LLM = factory.create_llm()
        self.embeddings = factory.create_embeddings()
