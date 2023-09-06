import configparser
from const import CONFIG_DATA_DIR, CONFIG_QA_CHAIN_TYPE, CONFIG_SETTING
from llm import LLM
from prompt_generator import PromptGenerator
from qa_chain import QAChain
from vectorstore import VectorStore

class Bot:
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        if CONFIG_SETTING not in config:
            raise ValueError(f'The configuration doesn\'t contain {CONFIG_SETTING} section')
        if CONFIG_DATA_DIR not in config[CONFIG_SETTING]:
            raise ValueError(f'The configuration\'s {CONFIG_SETTING} section doesn\'t contain {CONFIG_DATA_DIR}')
        if CONFIG_QA_CHAIN_TYPE not in config[CONFIG_SETTING]:
            raise ValueError(f'The configuration\'s {CONFIG_SETTING} section doesn\'t contain {CONFIG_QA_CHAIN_TYPE}')
        data_dir = config[CONFIG_SETTING][CONFIG_DATA_DIR]
        qa_chain_type = config[CONFIG_SETTING][CONFIG_QA_CHAIN_TYPE]
        llm = LLM(config)
        prompt_generator = PromptGenerator(config)
        vector_store = VectorStore(data_dir, llm)

        self.qa_chain = QAChain(qa_chain_type, llm, vector_store, prompt_generator)

    def run(self):
        while True:
            query = input(">")
            # transfer query to lower case
            query = query.lower().strip()
            if query in ['exit', 'quit', 'q', 'e', 'x']:
                break
            if not query:
                continue
            reply = self.qa_chain.ask(query)
            print(reply['output_text'])
