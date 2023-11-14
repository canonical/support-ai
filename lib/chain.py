from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from lib.datasources.ds_querier import DSQuerier

class Chain:
    def __init__(self, datasources):
        self.datasources = datasources
        self.ds_querier = DSQuerier(datasources)

    def ask(self, query, ds_type=None):
        resp = ''
        for content in self.ds_querier.query(query, ds_type):
            if not content:
                continue
            if resp:
                resp += '>>>>>>\n'
            resp += content
        return resp
