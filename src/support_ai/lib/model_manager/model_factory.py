from abc import ABC, abstractmethod


class ModelFactory(ABC):
    @abstractmethod
    def create_llm(self):
        return NotImplemented

    @abstractmethod
    def create_embeddings(self):
        return NotImplemented
