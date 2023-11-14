from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Data:
    Collection: str
    Document: str
    Metadata: dict
    Id: str

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
