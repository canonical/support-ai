from dataclasses import dataclass
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLLM
from lib.const import CONFIG_LLMS, CONFIG_NAME, CONFIG_TYPE, CONFIG_LLM, \
        CONFIG_EMBEDDINGS, CONFIG_HUGGINGFACE_PIPELINE, CONFIG_LLAMACPP, \
        CONFIG_OLLAMA, CONFIG_OPENAI
from lib.model_manager.huggingface_factory import HuggingFaceFactory
from lib.model_manager.llamacpp_factory import LlamaCppFactory
from lib.model_manager.ollama_factory import OllamaFactory
from lib.model_manager.openai_factory import OpenAIFactory


LLM_CONFIG = 'llm_config'
LLM_INST = 'llm_inst'
EMBEDDINGS_INST = 'embeddings_inst'


def get_model(llm_config):
    __factories = {
            CONFIG_HUGGINGFACE_PIPELINE: HuggingFaceFactory,
            CONFIG_LLAMACPP: LlamaCppFactory,
            CONFIG_OLLAMA: OllamaFactory,
            CONFIG_OPENAI: OpenAIFactory
            }
    if CONFIG_TYPE not in llm_config:
        raise ValueError(f'The llm config doesn\'t contain {CONFIG_TYPE}')
    if llm_config[CONFIG_TYPE] not in __factories:
        raise ValueError(f'Unknown llm type: {llm_config[CONFIG_TYPE]}')
    return __factories[llm_config[CONFIG_TYPE]](llm_config)

@dataclass
class Model:
    llm: BaseLLM
    embeddings: Embeddings

class ModelManager:
    __instance = None
    def __new__(cls, config):
        if cls.__instance is None:
            self = super(ModelManager, cls).__new__(cls)
            self.__models = {}
            if CONFIG_LLMS not in config:
                raise ValueError(f'The config doesn\'t contain {CONFIG_LLMS}')
            for llm in config[CONFIG_LLMS]:
                if CONFIG_NAME not in llm:
                    raise ValueError(f'The llm doesn\'t contain {CONFIG_NAME}')
                if llm[CONFIG_NAME] in self.__models:
                    raise ValueError(f'Duplicated llm name {llm[CONFIG_NAME]}')
                self.__models[llm[CONFIG_NAME]] = {
                        LLM_CONFIG: llm,
                        LLM_INST: None,
                        EMBEDDINGS_INST: None,
                        }
            cls.__instance = self
        return cls.__instance

    def get_model(self, config):
        model = Model(None, None)
        if CONFIG_LLM in config:
            if config[CONFIG_LLM] not in self.__models:
                raise ValueError(f'The llms doesn\'t contain a llm named "{config[CONFIG_LLM]}"')
            llm_name = config[CONFIG_LLM]
            if self.__models[llm_name][LLM_INST] is None:
                llm_config = self.__models[llm_name][LLM_CONFIG]
                self.__models[llm_name][LLM_INST] = get_model(llm_config).create_llm()
            model.llm = self.__models[llm_name][LLM_INST]

        if CONFIG_EMBEDDINGS in config:
            if config[CONFIG_EMBEDDINGS] not in self.__models:
                raise ValueError(f'The llms doesn\'t contain a llm named "{config[CONFIG_EMBEDDINGS]}"')
            embeddings_name = config[CONFIG_EMBEDDINGS]
            if self.__models[embeddings_name][EMBEDDINGS_INST] is None:
                embeddings_config = self.__models[embeddings_name][LLM_CONFIG]
                self.__models[embeddings_name][EMBEDDINGS_INST] = get_model(embeddings_config).create_embeddings()
            model.embeddings = self.__models[embeddings_name][EMBEDDINGS_INST]
        return model
