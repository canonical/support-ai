from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from lib.const import CONFIG_QA_CHAIN_TYPE, CONFIG_SETTING

class QAChain:
    def __init__(self, config, llm, ds_querier):
        if CONFIG_QA_CHAIN_TYPE not in config[CONFIG_SETTING]:
            raise ValueError(f'The configuration\'s {CONFIG_SETTING} ' +
                             f'section doesn\'t contain {CONFIG_QA_CHAIN_TYPE}')
        self.chain_type = config[CONFIG_SETTING][CONFIG_QA_CHAIN_TYPE]
        self.llm = llm
        self.ds_querier = ds_querier

    def ask(self, query):
        for prompt, content in self.ds_querier.query(query):
            prompt_tmpl = PromptTemplate.from_template(prompt)
            qa_chain = load_qa_chain(llm=self.llm.llm, chain_type=self.chain_type,
                                     prompt=prompt_tmpl)
            docs = []
            for data in content:
                docs.append(Document(page_content=data))
            result = qa_chain({'input_documents': docs}, return_only_outputs=True)
            print(result['output_text'])
