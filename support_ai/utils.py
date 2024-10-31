"""
This module provides functionality to load configuration data from YAML files.
"""

import pkgutil

import yaml
from support_ai.lib import const


def get_config(path):
    """
    Retrieve configuration data from a specified file or the default
    package resource.

    Args:
        path: The file path to the configuration YAML file. If None, the
              function will look for the default configuration file
              specified by `const.CONFIG_FILE` in the package.

    Returns:
        dict: The loaded configuration data as a dictionary.

    Raises:
        ValueError: If the default configuration file does not exist in the
                    package.
    """
    config = None
    if path is None:
        data = pkgutil.get_data(__package__, const.CONFIG_FILE)
        if data is None:
            raise ValueError(
                f'{const.CONFIG_FILE} doesn\'t exist in {__package__}')
        config = yaml.safe_load(data.decode('utf-8'))
    else:
        with open(path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    return config
