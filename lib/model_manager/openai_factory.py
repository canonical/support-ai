from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from lib.const import CONFIG_MODEL, CONFIG_LLM_OPENAI_API_KEY
from lib.model_manager.model_factory import ModelFactory


class OpenAIFactory(ModelFactory):
    def __init__(self, llm_config):
        self.model = llm_config[CONFIG_MODEL]
        self.api_key = llm_config[CONFIG_LLM_OPENAI_API_KEY]
        if not self.api_key:
            raise ValueError("Missing api_key in llm config")

    def create_llm(self):
        return OpenAI(
            openai_api_key=self.api_key,
            model_name=self.model,
        )

    def create_embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(model=self.model, openai_api_key=self.api_key)
