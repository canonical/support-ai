"""
This module provides the BaseContext class, which serves as a
foundational context for initializing and managing models using
a ModelManager instance.
"""

from .model_manager.model_manager import ModelManager


class BaseContext:  # pylint: disable=too-few-public-methods
    """
    BaseContext initializes the ModelManager with the given configuration.
    """

    def __init__(self, config):
        """
        Initializes the BaseContext with the given configuration.

        Args:
            config: Configuration dictionary used to set up
                    the ModelManager.
        """
        self.model_manager = ModelManager(config)
