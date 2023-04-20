from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import LlamaCpp, OpenAI


class LLM:

    def __init__(self, config):
        # get default language model (LLM) config
        default_llm = config.get('default_llm')
        llm_config = config.get(default_llm)

        # get model path, model name, and OpenAI API key
        model_path = llm_config.get('model_path')
        model_name = llm_config.get('model_name')
        openai_api_key = llm_config.get('openai_api_key')

        # create LLM and embedding based on LLM config
        if default_llm == 'llamacpp':
            if not model_path:
                raise ValueError("Missing model_path in config")

            self.llm = LlamaCpp(model_path=model_path, n_ctx=2048)
            self.embedding = LlamaCppEmbeddings(model_path=model_path, n_ctx=2048)

        elif default_llm == 'openai':
            if not openai_api_key:
                raise ValueError("Missing openai_api_key in config")

            self.llm = OpenAI(openai_api_key=openai_api_key, model_name=model_name)
            self.embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        else:
            raise ValueError("Unknown LLM type: {}".format(default_llm))