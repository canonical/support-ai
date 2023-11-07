from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from lib.const import CONFIG_QA_CHAIN_TYPE
from lib.datasources.ds_querier import DSQuerier

class QAChain:
    def __init__(self, config, datasources):
        if CONFIG_QA_CHAIN_TYPE not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_QA_CHAIN_TYPE}')
        self.chain_type = config[CONFIG_QA_CHAIN_TYPE]
        self.datasources = datasources
        self.ds_querier = DSQuerier(datasources)

    def __get_llm(self, ds_type):
        if ds_type not in self.datasources:
            raise ValueError(f'Unknown datasource type: {ds_type}')
        return self.datasources[ds_type].model_manager.llm

    def ask(self, query):
        for ds_type, prompt, content in self.ds_querier.query(query):
            prompt_tmpl = PromptTemplate.from_template(prompt)
            qa_chain = load_qa_chain(llm=self.__get_llm(ds_type),
                                     chain_type=self.chain_type,
                                     prompt=prompt_tmpl)
            docs = []
            for data in content:
                docs.append(Document(page_content=data))
            result = qa_chain({'input_documents': docs}, return_only_outputs=True)
            print(result['output_text'])
