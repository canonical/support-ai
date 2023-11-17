import simple_salesforce
from functools import partial
from operator import itemgetter
from langchain.callbacks.manager import trace_as_chain_group
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from lib.const import CONFIG_AUTHENTICATION, CONFIG_USERNAME, \
        CONFIG_PASSWORD, CONFIG_TOKEN
from lib.datasources.ds import Data, Content, Datasource
from lib.utils.lru import timed_lru_cache
from lib.model_manager import ModelManager


SYMPTOMS_PROMPT = """Generate five symptoms of the following:
    "{desc}"
    SYMPTOMS:"""
SUMMARY_PROMPT = """Summarize the following dialogs:
    "{comments}"
    SUMMARY:"""
REFINE_PROMPT = """Here's your first summary: {prev_summary}.
Now add to it based on the following context: {context}
"""
SEPERATOR = "%%%%%%%%"

def get_authentication(auth_config):
    if CONFIG_USERNAME not in auth_config:
        raise ValueError(f'The auth config doesn\'t contain {CONFIG_USERNAME}')
    if CONFIG_PASSWORD not in auth_config:
        raise ValueError(f'The auth config doesn\'t contain {CONFIG_PASSWORD}')
    if CONFIG_TOKEN not in auth_config:
        raise ValueError(f'The auth config doesn\'t contain {CONFIG_TOKEN}')
    return {
            'username': auth_config[CONFIG_USERNAME],
            'password': auth_config[CONFIG_PASSWORD],
            'security_token': auth_config[CONFIG_TOKEN]
            }

class SalesforceSource(Datasource):
    def __init__(self, config):
        if CONFIG_AUTHENTICATION not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_AUTHENTICATION}')
        auth = get_authentication(config[CONFIG_AUTHENTICATION])
        self.sf = simple_salesforce.Salesforce(**auth)
        self.model_manager = ModelManager(config)

    def __generate_symptoms(self, desc):
        prompt = PromptTemplate.from_template(SYMPTOMS_PROMPT)
        chain = (
                {'desc': RunnablePassthrough()}
                | prompt
                | self.model_manager.llm
                | StrOutputParser()
                )
        return chain.invoke(desc)

    def __get_cases(self, start_date=None, end_date=None):
        clause = ''
        conditions = []
        if start_date is not None:
            conditions.append(f'LastModifiedDate >= {start_date.isoformat()}T00:00:00Z')
        if end_date is not None:
            conditions.append(f'LastModifiedDate < {end_date.isoformat()}T00:00:00Z')

        for condition in conditions:
            if clause:
                clause += ' AND '
            clause += condition

        sql_cmd = 'SELECT Id, CaseNumber, Subject, Description ' + \
                'FROM Case' + (f' WHERE {clause}' if clause else '')
        cases = self.sf.query_all(sql_cmd)

        for case in cases['records']:
            yield Data(
                    self.__generate_symptoms(case['Description']),
                    {'id': case['Id'], 'subject': case['Subject']},
                    case['CaseNumber']
                    )

    def get_initial_data(self, start_date):
        return self.__get_cases(start_date)

    def get_update_data(self, start_date, end_date):
        return self.__get_cases(start_date, end_date)

    def __translate_comments(self, records):
        comments = []
        for comment in records:
            _records = self.sf.query_all(f'SELECT FirstName FROM User WHERE Id = \'{comment["CreatedById"]}\'')
            firstname = _records['records'][0]['FirstName']
            comments.append(f'{firstname}: {comment["CommentBody"]}')
        return SEPERATOR.join(comment for comment in comments)

    @timed_lru_cache()
    def __get_summary(self, comments):
        splitter = CharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=128,
                separator=SEPERATOR
                )
        docs = splitter.create_documents([comments])
        summary_prompt = PromptTemplate.from_template(SUMMARY_PROMPT)
        document_prompt = PromptTemplate.from_template("{page_content}")
        partial_format_doc = partial(format_document, prompt=document_prompt)
        summary_chain = (
                {'comments': partial_format_doc}
                | summary_prompt
                | self.model_manager.llm
                | StrOutputParser()
                )
        refine_prompt = PromptTemplate.from_template(REFINE_PROMPT)
        refine_chain = (
                {
                    'prev_summary': itemgetter('prev_summary'),
                    'context': lambda input: partial_format_doc(input['doc']),
                }
                | refine_prompt
                | self.model_manager.llm
                | StrOutputParser()
                )

        with trace_as_chain_group('refine loop', inputs={'input': docs}) as manager:
            summary = summary_chain.invoke(docs[0], config={'callbacks': manager})
            for doc in docs[1:]:
                summary = refine_chain.invoke(
                        {"prev_summary": summary, "doc": doc},
                        config={"callbacks": manager}
                        )
            manager.on_chain_end({"output": summary})
        return summary

    def get_content(self, metadata):
        cases = self.sf.query_all(f'SELECT Status, Public_Bug_URL__c, Sev_Lvl__c, CaseNumber FROM Case ' +
                                 f'WHERE Id = \'{metadata["id"]}\'')
        case = cases['records'][0]

        records = self.sf.query_all(f'SELECT CommentBody, CreatedById FROM CaseComment '
                                     f'WHERE ParentId = \'{metadata["id"]}\' AND IsPublished = True '
                                     f'ORDER BY LastModifiedDate')
        comments = self.__translate_comments(records['records'])
        return Content(
                {
                    'case_number': case['CaseNumber'],
                    'status': case['Status'],
                    'sev_lv': case['Sev_Lvl__c'],
                    'bug_url': case['Public_Bug_URL__c']
                },
                self.__get_summary(comments)
                )

    def generate_output(self, content):
        return f'Case:\t\t{content.Metadata["case_number"]}\n' \
                f'Status:\t\t{content.Metadata["status"]}\n' \
                f'Severity Level:\t{content.Metadata["sev_lv"]}\n' \
                f'Bug URL:\t{content.Metadata["bug_url"]}\n' \
                f'Summary:\n{content.Summary}\n'
