import time
import threading
import yaml
from .llm import LLM
from .prompt_generator import PromptGenerator
from .qa_chain import QAChain
from .vectorstore import VectorStore

stop_event = threading.Event()

class Bot:

    def __init__(self, CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        default_llm = config.get('default_llm')
        if default_llm is None:
            raise ValueError("default_llm is necessary; however, the config is [{}]".format(config))

        data_dir = config.get('data_dir')
        if data_dir is None:
            raise ValueError("data_dir is necessary; however, the config is [{}]".format(config))

        qa_chain_type = config.get('qa_chain_type', 'stuff')
        max_res_num = config.get('max_res_num', 3)

        self.llm = LLM(config)
        self.prompt_generator = PromptGenerator()
        self.vector_store = VectorStore(data_dir, self.llm)
        self.qa_chain = QAChain(qa_chain_type, max_res_num, self.llm, self.vector_store, self.prompt_generator)

    def update_vectordb(self):
        while not stop_event.is_set():
            self.vector_store.update()
            time.sleep(10)

    def run(self):
        update_vectordb_thread = threading.Thread(target=self.update_vectordb)

        stop_event.clear()
        update_vectordb_thread.start()
        while True:
            query = input(">")
            """transfer query to lower case"""
            query = query.lower().strip()
            if query in ['exit', 'quit', 'q', 'e', 'x']:
                break

            if not query:
                continue
            replies = self.qa_chain.ask(query)
            print("{}".format(replies))
        stop_event.set()
        update_vectordb_thread.join()
