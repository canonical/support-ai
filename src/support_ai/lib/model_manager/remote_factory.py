"""
This module contains classes for interacting with remote language models and
embeddings via HTTP requests.
"""

import json
from typing import Any, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from pydantic import BaseModel

from .. import const
from .model_factory import ModelFactory


class RemoteLLM(LLM):
    """
    A class representing a remote language model for generating text.
    """

    url: str
    token: str
    hostname: str

    @property
    def _llm_type(self) -> str:
        """
        Returns the type of the language model.
        """
        return "Remote"

    def _call(
        self,
        prompt: str,
        _stop: Optional[List[str]] = None,
        _run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Sends a prompt to the remote language model and retrieves the response.

        Args:
            prompt: The input prompt for the language model.
            stop: Optional stopping sequences.
            run_manager: An optional callback manager for LLM runs.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated text response from the language model.

        Raises:
            requests.HTTPError: If the request to the remote service fails.
        """
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
        """
        Returns identifying parameters for the remote model.

        Returns:
            Mapping[str, Any]: A dictionary of identifying parameters.
        """
        return {'url': self.url}


class RemoteEmbeddings(BaseModel, Embeddings):
    """
    A class for generating embeddings from a remote service.
    """

    url: str

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Sends a list of documents to the remote service and returns their
        embeddings.

        Args:
            texts: A list of text documents to embed.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is
                               a list of floats.

        Raises:
            requests.HTTPError: If the request to the remote service fails.
        """
        resp = requests.post(
                self.url, json.dumps({
                    'type': 'embeddings',
                    'texts': texts
                    })
                )
        resp.raise_for_status()
        return [list(map(float, output)) for output in resp.json()['outputs']]

    def embed_query(self, text: str) -> List[float]:
        """
        Sends a single query to the remote service and returns its embedding.

        Args:
            text: The text to embed.

        Returns:
            List[float]: The embedding of the input text.
        """
        return self.embed_documents([text])[0]


class RemoteFactory(ModelFactory):
    """
    A factory class for creating instances of RemoteLLM and RemoteEmbeddings.
    """

    def __init__(self, llm_config):
        """
        Initializes the RemoteFactory with configuration for remote LLM and
        embeddings.

        Args:
            llm_config: Configuration dictionary containing the URL, token,
                        and hostname.

        Raises:
            ValueError: If any required configuration parameter is missing.
        """
        self.url = llm_config[const.CONFIG_LLM_REMOTE_URL]
        if not self.url:
            raise ValueError(f'Missing {const.CONFIG_LLM_REMOTE_URL} in '
                             'llm config')
        self.token = llm_config[const.CONFIG_LLM_REMOTE_TOKEN]
        if not self.token:
            raise ValueError(f'Missing {const.CONFIG_LLM_REMOTE_TOKEN} in '
                             'llm config')
        self.hostname = llm_config[const.CONFIG_LLM_REMOTE_HOSTNAME]
        if not self.hostname:
            raise ValueError(f'Missing {const.CONFIG_LLM_REMOTE_HOSTNAME} in '
                             'llm config')

    def create_llm(self):
        """
        Creates an instance of the RemoteLLM.

        Returns:
            RemoteLLM: A configured instance of RemoteLLM.
        """
        return RemoteLLM(url=self.url, token=self.token,
                         hostname=self.hostname)

    def create_embeddings(self):
        """
        Creates an instance of RemoteEmbeddings.

        Returns:
            RemoteEmbeddings: A configured instance of RemoteEmbeddings.
        """
        return RemoteEmbeddings(url=self.url, token=self.token,
                                hostname=self.hostname)
