import hashlib
import glob
import os
import queue
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

VECTORDB_DIR = 'vectordb'
COLLECTION_METADATA = 'collection_metadata'
DEFAULT_COLLECTION_NAME = '__default'
BUF_SIZE = 4096

class VectorStore:
    def __init__(self, data_dir, llm):
        self.data_dir = data_dir
        self.llm = llm
        self.dbs = {}
        self.__construct()

    def __check_hash_change(self, collection_name, hash):
        meta_path = os.path.join(COLLECTION_METADATA, collection_name)
        if not os.path.exists(meta_path):
            return True
        with open(meta_path, 'r') as f:
            _hash = f.read()
        return hash != _hash

    def __store_metadata(self, collection_name, hash):
        meta_path = os.path.join(COLLECTION_METADATA, collection_name)
        with open(meta_path, 'w+') as f:
            f.write(hash)

    def __clear_stale_collections(self, collection_names):
        file_list = glob.glob(os.path.join(COLLECTION_METADATA, "*"))
        for file in file_list:
            collection_name = os.path.basename(file)
            if collection_name in collection_names:
                continue
            Chroma(collection_name=collection_name,
                   embedding_function=self.llm.embedding,
                   persist_directory=VECTORDB_DIR).delete_collection()
            os.remove(file)

    def __construct(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        collections = self.__get_collections()
        if not os.path.exists(COLLECTION_METADATA):
            os.mkdir(COLLECTION_METADATA)
        for collection_name, hash_docs in collections.items():
            if not self.__check_hash_change(collection_name, hash_docs[0]):
                self.dbs[collection_name] = Chroma(collection_name=collection_name,
                                                   embedding_function=self.llm.embedding,
                                                   persist_directory=VECTORDB_DIR)
                continue
            Chroma(collection_name=collection_name,
                   embedding_function=self.llm.embedding,
                   persist_directory=VECTORDB_DIR).delete_collection()
            docs = text_splitter.split_documents(hash_docs[1])
            self.dbs[collection_name] = Chroma.from_documents(collection_name=collection_name,
                                                              documents=docs,
                                                              embedding=self.llm.embedding,
                                                              persist_directory=VECTORDB_DIR)
            self.dbs[collection_name].persist()
            self.__store_metadata(collection_name, hash_docs[0])
        self.__clear_stale_collections(list(collections.keys()))

    def __calc_hash(self, file, hash):
        with open(file, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                hash.update(data)

    def __get_collections(self):
        collection_docs = {}
        q = queue.Queue()

        q.put(self.data_dir)
        while q.empty() is False:
            dir = q.get()
            collection_name = DEFAULT_COLLECTION_NAME if dir == self.data_dir else os.path.basename(dir)
            docs = []
            md5 = hashlib.md5()
            file_list = glob.glob(os.path.join(dir, "*"))
            for file in file_list:
                if os.path.isdir(file):
                    q.put(file)
                    continue
                self.__calc_hash(file, md5)
                docs.extend(TextLoader(file).load())
            if docs:
                collection_docs[collection_name] = (md5.hexdigest(), docs)
        return collection_docs
