from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from .. import const as const
from .model_factory import ModelFactory


class OllamaFactory(ModelFactory):
    def __init__(self, config):
        self.model = config[const.CONFIG_MODEL]
        if not self.model:
            raise ValueError(f'Missing {const.CONFIG_MODEL} in llm config')

    def create_llm(self):
        return ChatOllama(model=self.model)

    def create_embeddings(self):
        return OllamaEmbeddings(model=self.model)
