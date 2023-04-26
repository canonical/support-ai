from langchain.embeddings import LlamaCppEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import LlamaCpp, OpenAI, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def buildLlama(llm_config):
    model_path = llm_config.get('model_path')
    if not model_path:
        raise ValueError("Missing model_path in config")
    return LlamaCpp(model_path=model_path, n_ctx=2048), \
            LlamaCppEmbeddings(model_path=model_path, n_ctx=2048)

def buildOpenAI(llm_config):
    model_name = llm_config.get('model_name')
    openai_api_key = llm_config.get('openai_api_key')
    if not openai_api_key:
        raise ValueError("Missing openai_api_key in config")
    return OpenAI(openai_api_key=openai_api_key, model_name=model_name), \
            OpenAIEmbeddings(openai_api_key=openai_api_key)

def buildHuggingfacePipeline(llm_config):
    model_name = llm_config.get('model_name')
    if not model_name:
        raise ValueError("Missing model_name in config")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=120
    )
    return HuggingFacePipeline(pipeline=pipe), HuggingFaceEmbeddings(model_name=model_name)

class LLMFactory:
    @staticmethod
    def build (default_llm, llm_config):
        # create LLM and embedding based on LLM config
        match default_llm:
            case 'llamacpp':
                return buildLlama(llm_config)
            case 'openai':
                return buildOpenAI(llm_config)
            case 'huggingface_pipeline':
                return buildHuggingfacePipeline(llm_config)
            case _:
                raise ValueError("Unknown LLM type: {}".format(default_llm))

class LLM:
    def __init__(self, config):
        # get default language model (LLM) config
        default_llm = config.get('default_llm')
        llm_config = config.get(default_llm)

        self.llm, self.embedding = LLMFactory.build(default_llm, llm_config)
