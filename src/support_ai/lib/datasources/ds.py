from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Data:
    document: str
    metadata: dict
    id: str

@dataclass
class Content:
    metadata: dict
    summary: str

class Datasource(ABC):
    @abstractmethod
    def get_update_data(self, start_date, end_date):
        return NotImplemented

    @abstractmethod
    def get_content(self, doc):
        return NotImplemented

    @abstractmethod
    def generate_output(self, content):
        return NotImplemented
