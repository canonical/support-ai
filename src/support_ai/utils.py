import pkgutil

import yaml

from .lib import const


def get_config(path):
    config = None
    if path is None:
        data = pkgutil.get_data(__package__, const.CONFIG_FILE)
        if data is None:
            raise ValueError(
                f'{const.CONFIG_FILE} doesn\'t exist in {__package__}')
        config = yaml.safe_load(data.decode('utf-8'))
    else:
        with open(path) as stream:
            config = yaml.safe_load(stream)
    return config
