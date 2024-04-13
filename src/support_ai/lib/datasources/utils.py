from ..const import CONFIG_DATASOURCES, CONFIG_TYPE, CONFIG_SF, CONFIG_KB
from .kb import KnowledgeBaseSource
from .salesforce import SalesforceSource

_ds_mapping: dict = {
    CONFIG_SF: SalesforceSource,
    CONFIG_KB: KnowledgeBaseSource,
}

def get_datasources(config):
    datasources = {}
    if CONFIG_DATASOURCES not in config:
        raise ValueError(f'The config doesn\'t contain {CONFIG_DATASOURCES}')
    for ds in config[CONFIG_DATASOURCES]:
        if CONFIG_TYPE not in ds:
            raise ValueError(f'The datasource config doesn\'t contain {CONFIG_TYPE}')
        ds_type = ds[CONFIG_TYPE]

        if ds_type not in _ds_mapping:
            raise ValueError(f'Unknown datasource type: {ds_type}')
        datasources[ds_type] = _ds_mapping[ds_type](ds)
    return datasources
