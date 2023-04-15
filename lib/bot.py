from .vectorstore import VectorStore
from .llm import LLM

class Bot:
    def __init__(self, data_path, model_path):
        self.llm = LLM(model_path)
        self.vector_store = VectorStore(data_path, self.llm.embedding)

    def load_data(self):
        self.vector_store.load()

    def run(self):
        self.load_data()

        while True:
            query = input(">")
            reply = self.vector_store.search(query)
            print("{}".format(reply))
