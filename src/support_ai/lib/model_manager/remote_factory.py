import json
import requests
from typing import Any, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel
from ..const import CONFIG_LLM_REMOTE_HOSTNAME, CONFIG_LLM_REMOTE_TOKEN, CONFIG_LLM_REMOTE_URL
from .model_factory import ModelFactory


class RemoteLLM(LLM):
    url: str
    token: str
    hostname: str

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
        headers = {
                'Authorization': f'Bearer {self.token}',
                'Host': self.hostname,
                'Content-Type': 'application/json'
                }
        resp = requests.post(
                self.url, json.dumps({
                    'type': 'inference',
                    'texts': [prompt],
                    }),
                headers=headers
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
        self.token = llm_config[CONFIG_LLM_REMOTE_TOKEN]
        if not self.token:
            raise ValueError(f'Missing {CONFIG_LLM_REMOTE_TOKEN} in llm config')
        self.hostname = llm_config[CONFIG_LLM_REMOTE_HOSTNAME]
        if not self.hostname:
            raise ValueError(f'Missing {CONFIG_LLM_REMOTE_HOSTNAME} in llm config')

    def create_llm(self):
        return RemoteLLM(url=self.url, token=self.token, hostname=self.hostname)

    def create_embeddings(self):
        return RemoteEmbeddings(url=self.url, token=self.token, hostname=self.hostname)
