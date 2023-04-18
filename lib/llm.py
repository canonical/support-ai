from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.llms import LlamaCpp

class LLM:
    def __init__(self, model_path):
        self.llm = LlamaCpp(model_path=model_path, n_ctx=2048)
        self.embedding = LlamaCppEmbeddings(model_path=model_path, n_ctx=2048)
