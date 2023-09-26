import configparser
import lib.datasources.utils as ds_utils
from lib.const import CONFIG_SETTING
from lib.llm import LLM
from lib.datasources.ds_updater import DSUpdater
from lib.datasources.ds_querier import DSQuerier
from lib.qa_chain import QAChain
from lib.vectorstore import VectorStore

class Bot:
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        if CONFIG_SETTING not in config:
            raise ValueError(f'The configuration doesn\'t contain {CONFIG_SETTING} section')

        llm = LLM(config)
        vector_store = VectorStore(llm)
        datasources = ds_utils.get_datasources(config)
        ds_querier = DSQuerier(vector_store, datasources)

        self.ds_updater = DSUpdater(llm, vector_store, datasources)
        self.qa_chain = QAChain(config, llm, ds_querier)

    def run(self):
        self.ds_updater.initialize_data()
        while True:
            query = input(">")
            # transfer query to lower case
            query = query.lower().strip()
            if query in ['exit', 'quit', 'q', 'e', 'x']:
                break
            if not query:
                continue
            self.qa_chain.ask(query)
