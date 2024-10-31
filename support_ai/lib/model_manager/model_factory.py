"""
This module defines the ModelFactory abstract base class, which provides an
interface for creating language models (LLMs) and embeddings in various
implementations.
"""

from abc import ABC, abstractmethod


class ModelFactory(ABC):
    """
    Abstract base class for creating language models (LLMs) and embeddings.
    """

    @abstractmethod
    def create_llm(self):
        """
        Abstract method to create a language model instance.

        Returns:
            An instance of a language model.
        """
        return NotImplemented

    @abstractmethod
    def create_embeddings(self):
        """
        Abstract method to create an embeddings instance.

        Returns:
            An instance of embeddings.
        """
        return NotImplemented
