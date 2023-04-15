from langchain.embeddings.llamacpp import LlamaCppEmbeddings

class LLM:
    def __init__(self, model_path):
        self.embedding = LlamaCppEmbeddings(model_path=model_path)
