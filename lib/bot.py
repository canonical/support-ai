import time
import threading
from .llm import LLM
from .prompt_generator import PromptGenerator
from .qa_chain import QAChain
from .vectorstore import VectorStore

stop_event = threading.Event()

class Bot:
    def __init__(self, config):
        data_dir = config['data_dir']
        model_path = config['model_path']
        qa_chain_type = config['qa_chain_type'] if 'qa_chain_type' in config else 'stuff'
        max_res_num = config['max_res_num'] if 'max_res_num' in config else 3
        self.llm = LLM(model_path)
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
            if query == 'exit':
                break
            replies = self.qa_chain.ask(query)
            print("{}".format(replies))
        stop_event.set()
        update_vectordb_thread.join()
