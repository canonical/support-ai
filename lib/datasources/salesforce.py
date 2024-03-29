import simple_salesforce
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from lib.const import CONFIG_AUTHENTICATION, CONFIG_USERNAME, \
        CONFIG_PASSWORD, CONFIG_TOKEN
from lib.datasources.ds import Data, Content, Datasource
from lib.utils.docs_chain import docs_map_reduce, docs_refine
from lib.utils.lru import timed_lru_cache
from lib.model_manager import ModelManager


SYMPTOM_MAP_PROMPT = """Generate symptoms of the following:
    "{context}"
    SYMPTOMS:"""
SYMPTOM_REDUCE_PROMPT = """Combine the symptoms:
    "{context}"
    SYMPTOMS:"""
CONDENSE_INITIAL_PROMPT = """Summarize the following dialog:
    "{context}"
    SUMMARY:"""
CONDENSE_REFINE_PROMPT = """Here's the previous summary:
    "{prev_context}"
    Integrate it with the following dialog:
    "{context}"
    SUMMARY:"""
TROUBLESHOOTING_PROCESS_INITIAL_PROMPT = """Integrate the following conversation:
    "{context}"
    CONVERSATION:"""
TROUBLESHOOTING_PROCESS_REFINE_PROMPT = """Here's the previous integrated conversations:
    "{prev_context}"
    Integrate it with the following conversation:
    "{context}"
    CONVERSATION:"""
SOLUTION_JUDGEMENT_PROMPT = """Judge if the following comment has described root cause, solution or workaround:
    The issue is mainly caused by the bug. // YES
    The issue can be solved by the following approach. // YES
    We can provide a workaround to bypass this issue. // YES
    We are still working on this issue. // NO
    "{context}" //
"""
SOLUTION_INITIAL_PROMPT = """Extract the root cause, workaround or solution from the following conversation:
    "{context}"
    SOLUTION:"""
SOLUTION_REFINE_PROMPT = """Here's the previous extracted context:
    "{prev_context}"
    Combine it with the following conversation:
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


class SalesforceSource(Datasource):
    def __init__(self, config):
        if CONFIG_AUTHENTICATION not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_AUTHENTICATION}')
        auth = get_authentication(config[CONFIG_AUTHENTICATION])
        self.sf = simple_salesforce.Salesforce(**auth)
        self.model_manager = ModelManager(config)

    def __get_symptom(self, desc):
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=128,
                )
        docs = splitter.create_documents([desc])
        return docs_map_reduce(self.model_manager.llm, docs, SYMPTOM_MAP_PROMPT, SYMPTOM_REDUCE_PROMPT)

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
                    {'id': case['Id'], 'subject': case['Subject']},
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
                chunk_size=1536,
                chunk_overlap=128,
                length_function=len,
                )
        docs = splitter.create_documents(comments)
        return docs_refine(self.model_manager.llm, docs, CONDENSE_INITIAL_PROMPT, CONDENSE_REFINE_PROMPT)

    def __get_troubleshooting_process(self, dialogs):
        docs = []
        for dialog in dialogs:
            docs.append(Document(page_content=dialog['comment'], metadata={"user": dialog['user']}))
        return docs_refine(self.model_manager.llm, docs, TROUBLESHOOTING_PROCESS_INITIAL_PROMPT, TROUBLESHOOTING_PROCESS_REFINE_PROMPT)

    def __get_solution(self, dialogs):
        prompt = PromptTemplate.from_template(SOLUTION_JUDGEMENT_PROMPT)
        chain = (
                {'context': RunnablePassthrough()}
                | prompt
                | self.model_manager.llm
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
        return docs_refine(self.model_manager.llm, docs, SOLUTION_INITIAL_PROMPT, SOLUTION_REFINE_PROMPT)

    @timed_lru_cache()
    def __get_summary(self, desc, dialogs):
        condensed_dialogs = Dialogs()
        user = ''
        comments = []
        for dialog in dialogs:
            if not user or user == dialog['user']:
                user = dialog['user']
                comments.append(dialog['comment'])
                continue
            condensed_dialogs.append(user, self.__condense_comments(comments))
            user = dialog['user']
            comments = [dialog['comment']]
        condensed_dialogs.append(user, self.__condense_comments(comments))
        return '\n'.join([
            self.__get_symptom(desc),
            self.__get_troubleshooting_process(condensed_dialogs),
            self.__get_solution(condensed_dialogs)
            ])

    def get_content(self, metadata):
        case = self.sf.query_all(f'SELECT Status, Public_Bug_URL__c, Sev_Lvl__c, CaseNumber, Description FROM Case ' +
                                 f'WHERE Id = \'{metadata["id"]}\'')['records'][0]
        records = self.sf.query_all(f'SELECT CommentBody, CreatedById FROM CaseComment '
                                     f'WHERE ParentId = \'{metadata["id"]}\' AND IsPublished = True '
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

    def generate_output(self, content):
        return f'Case:\t\t{content.Metadata["case_number"]}\n' \
                f'Status:\t\t{content.Metadata["status"]}\n' \
                f'Severity Level:\t{content.Metadata["sev_lv"]}\n' \
                f'Bug URL:\t{content.Metadata["bug_url"]}\n' \
                f'Summary:\n{content.Summary}\n'
