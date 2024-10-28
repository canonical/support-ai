from .. import const
from .kb import KnowledgeBaseSource
from .salesforce import SalesforceSource

_ds_mapping: dict = {
    const.CONFIG_SF: SalesforceSource,
    const.CONFIG_KB: KnowledgeBaseSource,
}


def get_datasources(config):
    datasources = {}
    if const.CONFIG_DATASOURCES not in config:
        raise ValueError(f'The config doesn\'t contain '
                         f'{const.CONFIG_DATASOURCES}')
    for ds in config[const.CONFIG_DATASOURCES]:
        if const.CONFIG_TYPE not in ds:
            raise ValueError(f'The datasource config doesn\'t contain '
                             f'{const.CONFIG_TYPE}')
        ds_type = ds[const.CONFIG_TYPE]

        if ds_type not in _ds_mapping:
            raise ValueError(f'Unknown datasource type: {ds_type}')
        datasources[ds_type] = _ds_mapping[ds_type](ds)
    return datasources
