import yaml
from llm import LLM
from prompt_generator import PromptGenerator
from qa_chain import QAChain
from vectorstore import VectorStore

class Bot:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding="utf8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        default_llm = config.get('default_llm')
        if default_llm is None:
            raise ValueError(f'default_llm is necessary; however, the config is [{config}]')
        data_dir = config.get('data_dir')
        if data_dir is None:
            raise ValueError(f'data_dir is necessary; however, the config is [{config}]')
        qa_chain_type = config.get('qa_chain_type', 'stuff')

        self.llm = LLM(config)
        self.prompt_generator = PromptGenerator(config)
        self.vector_store = VectorStore(data_dir, self.llm)
        self.qa_chain = QAChain(qa_chain_type, self.llm, self.vector_store, self.prompt_generator)

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
