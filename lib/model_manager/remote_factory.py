import json
import requests
from typing import Any, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel
from lib.const import CONFIG_LLM_REMOTE_URL
from lib.model_manager.model_factory import ModelFactory


class RemoteLLM(LLM):
    url: str

    @property
    def _llm_type(self) -> str:
        return "Remote"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        resp = requests.post(
                self.url, json.dumps({
                    'type': 'inference',
                    'texts': [prompt],
                    })
                )
        resp.raise_for_status()
        return resp.json()['outputs'][0]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'url': self.url}


class RemoteEmbeddings(BaseModel, Embeddings):
    url: str

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(
                self.url, json.dumps({
                    'type': 'embeddings',
                    'texts': texts
                    })
                )
        resp.raise_for_status()
        return [list(map(float, output)) for output in resp.json()['outputs']]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class RemoteFactory(ModelFactory):
    def __init__(self, llm_config):
        self.url = llm_config[CONFIG_LLM_REMOTE_URL]
        if not self.url:
            raise ValueError(f'Missing {CONFIG_LLM_REMOTE_URL} in llm config')

    def create_llm(self):
        return RemoteLLM(url=self.url)

    def create_embeddings(self):
        return RemoteEmbeddings(url=self.url)
