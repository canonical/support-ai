import time
import threading
from .vectorstore import VectorStore
from .llm import LLM

stop_event = threading.Event()

class Bot:
    def __init__(self, data_dir, model_path, qa_chain_type):
        self.llm = LLM(model_path)
        self.vector_store = VectorStore(data_dir, qa_chain_type, self.llm)

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
            reply = self.vector_store.search(query)
            print("{}".format(reply))
        stop_event.set()
        update_vectordb_thread.join()
