from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from .. import const
from .model_factory import ModelFactory


class HuggingFaceFactory(ModelFactory):
    def __init__(self, llm_config):
        self.model = llm_config[const.CONFIG_MODEL]
        if not self.model:
            raise ValueError(f'Missing {const.CONFIG_MODEL} in llm config')

    def create_llm(self):
        return HuggingFacePipeline.from_model_id(model_id=self.model,
                                                 task='text-generation')

    def create_embeddings(self):
        return HuggingFaceEmbeddings(model_name=self.model)
