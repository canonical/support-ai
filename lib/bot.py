import configparser
import lib.datasources.utils as ds_utils
from lib.const import CONFIG_QA_CHAIN_TYPE, CONFIG_SETTING
from lib.llm import LLM
from lib.datasources.ds_updater import DSUpdater
from lib.datasources.ds_querier import DSQuerier
from lib.vectorstore import VectorStore

class Bot:
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        if CONFIG_SETTING not in config:
            raise ValueError(f'The configuration doesn\'t contain {CONFIG_SETTING} section')
        if CONFIG_QA_CHAIN_TYPE not in config[CONFIG_SETTING]:
            raise ValueError(f'The configuration\'s {CONFIG_SETTING} ' +
                             f'section doesn\'t contain {CONFIG_QA_CHAIN_TYPE}')
        llm = LLM(config)
        vector_store = VectorStore(llm)
        datasources = ds_utils.get_datasources(config)

        self.ds_updater = DSUpdater(vector_store, datasources)
        self.ds_querier = DSQuerier(vector_store, datasources)

    def run(self):
        self.ds_updater.start_update()
        while True:
            query = input(">")
            # transfer query to lower case
            query = query.lower().strip()
            if query in ['exit', 'quit', 'q', 'e', 'x']:
                break
            if not query:
                continue
            self.ds_querier.query(query)
