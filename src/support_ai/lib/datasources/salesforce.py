import re
import simple_salesforce
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..const import CONFIG_AUTHENTICATION, CONFIG_USERNAME, \
        CONFIG_PASSWORD, CONFIG_TOKEN
from ..context import BaseContext
from ..utils.docs_chain import docs_refine
from ..utils.lru import timed_lru_cache
from .ds import Data, Content, Datasource


SYMPTOM_INITIAL_PROMPT = """Summarize symptom of the following content:
    "{context}"
    SUMMARY:"""
SYMPTOM_REFINE_PROMPT = """Here's the previous summary:
    "{prev_context}"
    Summarize symptom again with the following content
    "{context}"
    SUMMARY:"""
CONDENSE_INITIAL_PROMPT = """Summarize the following dialog in detail:
    "{context}"
    SUMMARY:"""
CONDENSE_REFINE_PROMPT = """Here's the previous summary:
    "{prev_context}"
    Summarize again with the following dialog in detail:
    "{context}"
    SUMMARY:"""
SOL_JUDGEMENT_PROMPT = """Judge if the following comment has described root cause, solution or workaround.
    The ANS is either "YES" or "NO".

    CONTEXT: "The issue is mainly caused by the bug."
    ANS: YES

    CONTEXT: "The issue can be solved by the following approach."
    ANS: YES

    CONTEXT: "We can provide a workaround to bypass this issue."
    ANS: YES

    CONTEXT: "We are still working on this issue."
    ANS: NO

    CONTEXT: "{context}"
    ANS: """
SOL_INITIAL_PROMPT = """Extract the root cause, workaround or solution from the following content in detail:
    "{context}"
    SOLUTION:"""
SOL_REFINE_PROMPT = """Here's the previous extracted content:
    "{prev_context}"
    Combine it with the following content in detail:
    "{context}"
    SOLUTION:"""


class Dialogs:
    def __init__(self):
        self.dialogs = []

    def append(self, user, comment):
        self.dialogs.append({
            'user': user,
            'comment': comment,
            })

    def __hash__(self):
        return hash(tuple(str(dialog) for dialog in self.dialogs))

    def __iter__(self):
        return iter(self.dialogs)


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


class SalesforceSource(BaseContext, Datasource):
    def __init__(self, config):
        super().__init__(config)
        if CONFIG_AUTHENTICATION not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_AUTHENTICATION}')
        auth = get_authentication(config[CONFIG_AUTHENTICATION])
        self.sf = simple_salesforce.Salesforce(**auth)
        self.model = self.model_manager.get_model(config)

    def __get_symptom(self, desc):
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=128,
                )
        docs = splitter.create_documents([desc])
        return docs_refine(self.model.llm, docs, SYMPTOM_INITIAL_PROMPT, SYMPTOM_REFINE_PROMPT)

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
        for case in self.sf.query_all(sql_cmd)['records']:
            if case['Description'] is None:
                continue
            yield Data(
                    self.__get_symptom(case['Description']),
                    {'case_number': case['CaseNumber'], 'subject': case['Subject']},
                    case['CaseNumber']
                    )

    def get_update_data(self, start_date, end_date):
        return self.__get_cases(start_date, end_date)

    def __translate_into_dialogs(self, records):
        dialogs = Dialogs()
        for comment in records:
            _records = self.sf.query_all(f'SELECT FirstName FROM User WHERE Id = \'{comment["CreatedById"]}\'')
            firstname = _records['records'][0]['FirstName']
            dialogs.append(firstname, comment["CommentBody"])
        return dialogs

    def __condense_comments(self, comments):
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=2048,
                chunk_overlap=128,
                length_function=len,
                )
        docs = splitter.create_documents(comments)
        return docs_refine(self.model.llm, docs, CONDENSE_INITIAL_PROMPT, CONDENSE_REFINE_PROMPT)

    def __get_process(self, dialogs):
        condensed_comments = []
        user = ''
        comments = []
        for dialog in dialogs:
            if not user or user == dialog['user']:
                user = dialog['user']
                comments.append(dialog['comment'])
                continue
            condensed_comments.append(self.__condense_comments(comments))
            user = dialog['user']
            comments = [dialog['comment']]
        condensed_comments.append(self.__condense_comments(comments))
        process_stmt = ' '.join(condensed_comments).replace('\n', '')
        return re.sub('\s+', ' ', process_stmt).strip()

    def __get_solution(self, dialogs):
        prompt = PromptTemplate.from_template(SOL_JUDGEMENT_PROMPT)
        chain = (
                {'context': RunnablePassthrough()}
                | prompt
                | self.model.llm
                | StrOutputParser()
                )
        comments = []
        for dialog in dialogs:
            result = chain.invoke(dialog['comment'])
            if result != 'NO':
                comments.append(dialog['comment'])

        docs = []
        for comment in comments:
            docs.append(Document(page_content=comment))
        return docs_refine(self.model.llm, docs, SOL_INITIAL_PROMPT, SOL_REFINE_PROMPT)

    @timed_lru_cache()
    def __get_summary(self, desc, dialogs):
        return '\n'.join([
            self.__get_symptom(desc),
            self.__get_process(dialogs),
            self.__get_solution(dialogs)
            ])

    def __get_content(self, case_number):
        case = self.sf.query_all(f'SELECT Id, Status, Public_Bug_URL__c, Sev_Lvl__c, CaseNumber, Description FROM Case ' +
                                 f'WHERE CaseNumber = \'{case_number}\'')['records'][0]
        records = self.sf.query_all(f'SELECT CommentBody, CreatedById FROM CaseComment '
                                     f'WHERE ParentId = \'{case["Id"]}\' AND IsPublished = True '
                                     f'ORDER BY LastModifiedDate')
        dialogs = self.__translate_into_dialogs(records['records'])
        return Content(
                {
                    'case_number': case['CaseNumber'],
                    'status': case['Status'],
                    'sev_lv': case['Sev_Lvl__c'],
                    'bug_url': case['Public_Bug_URL__c']
                },
                self.__get_summary(case['Description'], dialogs)
                )

    def get_content(self, metadata):
        return self.__get_content(metadata['case_number'])

    def generate_output(self, content):
        return f'Case:\t\t{content.metadata["case_number"]}\n' \
                f'Status:\t\t{content.metadata["status"]}\n' \
                f'Severity Level:\t{content.metadata["sev_lv"]}\n' \
                f'Bug URL:\t{content.metadata["bug_url"]}\n' \
                f'Summary:\n{content.summary}\n'

    def summarize_case(self, case_number):
        return self.__get_content(case_number)
