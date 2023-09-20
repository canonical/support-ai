from lib.const import CONFIG_DATASOURCES, CONFIG_SF, CONFIG_SETTING
from lib.datasources.salesforce import SalesforceSource

_datasources: dict = {
    CONFIG_SF: SalesforceSource
}

def get_datasources(config):
    datasources = {}
    if CONFIG_DATASOURCES not in config[CONFIG_SETTING]:
        raise ValueError(f'The configuration\'s {CONFIG_SETTING} ' +
                         f'section doesn\'t contain {CONFIG_DATASOURCES}')

    for ds_type in config.get(CONFIG_SETTING, CONFIG_DATASOURCES).split('|'):
        if ds_type not in config:
            print(f'The configuration doesn\'t contain {ds_type} section')
            continue
        if ds_type not in _datasources:
            raise ValueError(f'Unknown datasource: {ds_type}')
        datasources[ds_type] = _datasources[ds_type](config[ds_type])
    return datasources
