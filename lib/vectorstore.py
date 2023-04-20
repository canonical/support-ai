import glob
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

VECTORDB_DIR = 'db'

class VectorStore:
    def __init__(self, data_dir, llm):
        self.data_dir = data_dir
        docs, file_list = self.__get_docs()
        if docs:
            self.db = Chroma.from_documents(documents=docs, embedding=llm.embedding, persist_directory=VECTORDB_DIR)
            self.db.persist()
            self.__remove_files(file_list)
        else:
            self.db = Chroma(embedding_function=llm.embedding, persist_directory=VECTORDB_DIR)

    def __del__(self):
        self.db.persist()

    def __get_docs(self):
        documents = []
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        file_list = glob.glob(os.path.join(self.data_dir, "*.data"))
        for file in file_list:
            documents.extend(TextLoader(file).load())
        return text_splitter.split_documents(documents), file_list

    def __remove_files(self, file_list):
        for file in file_list:
            os.remove(file)

    def update(self):
        docs, file_list = self.__get_docs()
        if not docs:
            return
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        self.db.add_texts(texts=texts, metadatas=metadatas)
        self.__remove_files(file_list)
