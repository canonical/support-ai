from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from ..const import CONFIG_MODEL
from .model_factory import ModelFactory


class LlamaCppFactory(ModelFactory):
    def __init__(self, llm_config):
        self.model = llm_config[CONFIG_MODEL]
        if not self.model:
            raise ValueError(f'Missing {CONFIG_MODEL} in llm config')

    def create_llm(self):
        return LlamaCpp(model_path=self.model, n_ctx=4096)

    def create_embeddings(self):
        return LlamaCppEmbeddings(model_path=self.model, n_ctx=4096)
