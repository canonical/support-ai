from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from lib.const import CONFIG_QA_CHAIN_TYPE
from lib.datasources.ds_querier import DSQuerier
from lib.lru import timed_lru_cache

class QAChain:
    def __init__(self, config, datasources):
        if CONFIG_QA_CHAIN_TYPE not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_QA_CHAIN_TYPE}')
        self.chain_type = config[CONFIG_QA_CHAIN_TYPE]
        self.datasources = datasources
        self.ds_querier = DSQuerier(datasources)

    def __get_ds(self, ds_type):
        if ds_type not in self.datasources:
            raise ValueError(f'Unknown datasource type: {ds_type}')
        return self.datasources[ds_type]

    @timed_lru_cache()
    def __parse_body(self, ds, prompt, body):
        if not prompt:
            return body

        prompt_tmpl = PromptTemplate.from_template(prompt)
        qa_chain = load_qa_chain(llm=ds.model_manager.llm,
                                 chain_type=self.chain_type,
                                 prompt=prompt_tmpl)
        docs = [Document(page_content=body)]
        result = qa_chain({'input_documents': docs}, return_only_outputs=True)
        return result['output_text']

    def ask(self, query, ds_type=None):
        resp = ''
        for _ds_type, raw_content in self.ds_querier.query(query, ds_type):
            ds = self.__get_ds(_ds_type)
            raw_content.Body = self.__parse_body(ds, raw_content.Prompt, raw_content.Body)
            content = ds.generate_content(raw_content)
            if not content:
                continue
            if resp:
                resp += '>>>>>>\n'
            resp += content
        return resp
