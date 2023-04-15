from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

class VectorStore:
    def __init__(self, data_path, embedding):
        loader = TextLoader(data_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        self.db = Chroma.from_documents(docs, embedding)

    def search(self, query):
        docs = self.db.similarity_search(query)
        return docs[0].page_content
