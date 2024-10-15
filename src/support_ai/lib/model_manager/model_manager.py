from dataclasses import dataclass
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLLM
from .. import const as const
from .huggingface_factory import HuggingFaceFactory
from .llamacpp_factory import LlamaCppFactory
from .ollama_factory import OllamaFactory
from .openai_factory import OpenAIFactory
from .remote_factory import RemoteFactory


LLM_CONFIG = 'llm_config'
LLM_INST = 'llm_inst'
EMBEDDINGS_INST = 'embeddings_inst'


def get_model(llm_config):
    __factories = {
            const.CONFIG_HUGGINGFACE_PIPELINE: HuggingFaceFactory,
            const.CONFIG_LLAMACPP: LlamaCppFactory,
            const.CONFIG_OLLAMA: OllamaFactory,
            const.CONFIG_OPENAI: OpenAIFactory,
            const.CONFIG_REMOTE: RemoteFactory,
            }
    if const.CONFIG_TYPE not in llm_config:
        raise ValueError(f'The llm config doesn\'t contain {const.CONFIG_TYPE}')
    if llm_config[const.CONFIG_TYPE] not in __factories:
        raise ValueError(f'Unknown llm type: {llm_config[const.CONFIG_TYPE]}')
    return __factories[llm_config[const.CONFIG_TYPE]](llm_config)

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
            if const.CONFIG_LLMS not in config:
                raise ValueError(f'The config doesn\'t contain {const.CONFIG_LLMS}')
            for llm in config[const.CONFIG_LLMS]:
                if const.CONFIG_NAME not in llm:
                    raise ValueError(f'The llm doesn\'t contain {const.CONFIG_NAME}')
                if llm[const.CONFIG_NAME] in self.__models:
                    raise ValueError(f'Duplicated llm name {llm[const.CONFIG_NAME]}')
                self.__models[llm[const.CONFIG_NAME]] = {
                        LLM_CONFIG: llm,
                        LLM_INST: None,
                        EMBEDDINGS_INST: None,
                        }
            cls.__instance = self
        return cls.__instance

    def get_model(self, config):
        model = Model(None, None)
        if const.CONFIG_LLM in config:
            if config[const.CONFIG_LLM] not in self.__models:
                raise ValueError(f'The llms doesn\'t contain a llm named "{config[const.CONFIG_LLM]}"')
            llm_name = config[const.CONFIG_LLM]
            if self.__models[llm_name][LLM_INST] is None:
                llm_config = self.__models[llm_name][LLM_CONFIG]
                self.__models[llm_name][LLM_INST] = get_model(llm_config).create_llm()
            model.llm = self.__models[llm_name][LLM_INST]

        if const.CONFIG_EMBEDDINGS in config:
            if config[const.CONFIG_EMBEDDINGS] not in self.__models:
                raise ValueError(f'The llms doesn\'t contain a llm named "{config[const.CONFIG_EMBEDDINGS]}"')
            embeddings_name = config[const.CONFIG_EMBEDDINGS]
            if self.__models[embeddings_name][EMBEDDINGS_INST] is None:
                embeddings_config = self.__models[embeddings_name][LLM_CONFIG]
                self.__models[embeddings_name][EMBEDDINGS_INST] = get_model(embeddings_config).create_embeddings()
            model.embeddings = self.__models[embeddings_name][EMBEDDINGS_INST]
        return model
