from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Data:
    Document: str
    Metadata: dict
    Id: str

@dataclass
class Content:
    Metadata: dict
    Summary: str

class Datasource(ABC):
    @abstractmethod
    def get_initial_data(self, start_date):
        return NotImplemented

    @abstractmethod
    def get_update_data(self, start_date, end_date):
        return NotImplemented

    @abstractmethod
    def get_content(self, doc):
        return NotImplemented

    @abstractmethod
    def generate_output(self, content):
        return NotImplemented
