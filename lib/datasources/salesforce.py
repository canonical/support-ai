import simple_salesforce
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from lib.const import CONFIG_AUTHENTICATION, CONFIG_USERNAME, \
        CONFIG_PASSWORD, CONFIG_TOKEN
from lib.datasources.ds import Data, Content, Datasource
from lib.lru import timed_lru_cache
from lib.model_manager import ModelManager


SYMPTOMS_PROMPT = """Generate five symptoms of the following:
    "{desc}"
    SYMPTOMS:"""
SUMMARY_PROMPT = """Write a concise summary of the following:
    "{comments}"
    CONCISE SUMMARY:"""

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

    @timed_lru_cache()
    def __get_summary(self, comments):
        prompt = PromptTemplate.from_template(SUMMARY_PROMPT)
        chain = (
                {'comments': RunnablePassthrough()}
                | prompt
                | self.model_manager.llm
                | StrOutputParser()
                )
        return chain.invoke(comments)

    def get_content(self, doc):
        cases = self.sf.query_all(f'SELECT Status, Public_Bug_URL__c, Sev_Lvl__c, CaseNumber FROM Case ' +
                                 f'WHERE Id = \'{doc.metadata["id"]}\'')
        case = cases['records'][0]
        metadata = {
                'case_number': case['CaseNumber'],
                'status': case['Status'],
                'sev_lv': case['Sev_Lvl__c'],
                'bug_url': case['Public_Bug_URL__c']
                }

        comments = self.sf.query_all(f'SELECT CommentBody FROM CaseComment ' 
                                     f'WHERE ParentId = \'{doc.metadata["id"]}\' '
                                     f'ORDER BY LastModifiedDate')
        body = ''
        for comment in comments['records']:
            if body:
                body += '\n'
            body += comment['CommentBody']
        return Content(
                metadata,
                self.__get_summary(body)
                )

    def generate_output(self, content):
        return f'Case:\t\t{content.Metadata["case_number"]}\n' \
                f'Status:\t\t{content.Metadata["status"]}\n' \
                f'Severity Level:\t{content.Metadata["sev_lv"]}\n' \
                f'Bug URL:\t{content.Metadata["bug_url"]}\n' \
                f'Summary:\n{content.Summary}\n'
