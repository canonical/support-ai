"""
Module for defining data structures and an abstract base class for
data sources.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Data:
    """
    Represents a document with associated metadata and an identifier.
    """

    document: str
    metadata: dict
    id: str


@dataclass
class Content:
    """
    Represents content containing metadata and a summary.
    """

    metadata: dict
    summary: str


class Datasource(ABC):
    """
    Abstract base class for data source implementations.
    """

    @abstractmethod
    def get_update_data(self, start_date, end_date):
        """
        Retrieve updated data based on the specified date range.

        Args:
            start_date: The start date for filtering updates.
            end_date: The end date for filtering updates.

        Returns:
            NotImplemented: Must be overridden by subclasses.
        """
        return NotImplemented

    @abstractmethod
    def get_content(self, metadata):
        """
        Retrieve content based on the provided metadata.

        Args:
            metadata: The metadata used to retrieve the content.

        Returns:
            NotImplemented: Must be overridden by subclasses.
        """
        return NotImplemented

    @abstractmethod
    def custom_api(self, action, data):
        """
        Handle custom API actions based on the provided action and data.

        Args:
            action: The action to perform.
            data: The data associated with the action.

        Returns:
            NotImplemented: Must be overridden by subclasses.
        """
        return NotImplemented

    @abstractmethod
    def generate_output(self, content):
        """
        Generate output based on the provided content.

        Args:
            content: The content used to generate the output.

        Returns:
            NotImplemented: Must be overridden by subclasses.
        """
        return NotImplemented
