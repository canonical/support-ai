"""
This module provides functionality to interact with Salesforce.
"""

import re

import simple_salesforce
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import const
from ..context import BaseContext
from ..utils.docs_chain import docs_refine
from ..utils.lru import timed_lru_cache
from ..utils.parallel_executor import run_fn_in_parallel, run_in_parallel
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
SOL_JUDGEMENT_PROMPT = \
"""Judge if the following comment has described root cause, solution or workaround.
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
    ANS: """  # noqa
SOL_INITIAL_PROMPT = \
"""Extract the root cause, workaround or solution from the following content in detail:
    "{context}"
    SOLUTION:"""  # noqa
SOL_REFINE_PROMPT = """Here's the previous extracted content:
    "{prev_context}"
    Combine it with the following content in detail:
    "{context}"
    SOLUTION:"""


class Dialogs:
    """
    Class to represent and manage a collection of dialog entries.
    """

    def __init__(self):
        self.dialogs = []

    def append(self, user, comment):
        """
        Append a user comment to the dialog list.

        Args:
            user: The username of the commenter.
            comment: The comment made by the user.
        """
        self.dialogs.append({
            'user': user,
            'comment': comment,
            })

    def __hash__(self):
        """
        Return a hash of the dialogs for use in sets and dicts.
        """
        return hash(tuple(str(dialog) for dialog in self.dialogs))

    def __iter__(self):
        """
        Return an iterator over the dialog entries.
        """
        return iter(self.dialogs)


def get_authentication(auth_config):
    """
    Extract and validate authentication details from the configuration.

    Args:
        auth_config: The authentication configuration dictionary.

    Raises:
        ValueError: If any required authentication keys are missing.

    Returns:
        dict: A dictionary containing validated authentication details.
    """
    if const.CONFIG_USERNAME not in auth_config:
        raise ValueError(
            f'The auth config doesn\'t contain {const.CONFIG_USERNAME}')
    if const.CONFIG_PASSWORD not in auth_config:
        raise ValueError(
            f'The auth config doesn\'t contain {const.CONFIG_PASSWORD}')
    if const.CONFIG_TOKEN not in auth_config:
        raise ValueError(
            f'The auth config doesn\'t contain {const.CONFIG_TOKEN}')
    return {
            'username': auth_config[const.CONFIG_USERNAME],
            'password': auth_config[const.CONFIG_PASSWORD],
            'security_token': auth_config[const.CONFIG_TOKEN]
            }


class SalesforceSource(BaseContext, Datasource):
    """
    A source class for interacting with Salesforce to retrieve and
    process case data.
    """

    def __init__(self, config):
        """
        Initializes the SalesforceSource with the given configuration.

        Args:
            config: Configuration dictionary containing authentication
                details and other settings.

        Raises:
            ValueError: If the authentication configuration is missing.
        """
        super().__init__(config)
        if const.CONFIG_AUTHENTICATION not in config:
            raise ValueError(
                f'The config doesn\'t contain {const.CONFIG_AUTHENTICATION}')
        auth = get_authentication(config[const.CONFIG_AUTHENTICATION])
        self.sf = simple_salesforce.Salesforce(**auth)
        self.model = self.model_manager.get_model(config)

    def __get_symptom(self, desc):
        """
        Extracts symptoms from the provided case description.

        Args:
            desc: The case description from which to extract symptoms.

        Returns:
            List[Document]: A list of documents containing refined symptoms.
        """
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=128,
                )
        docs = splitter.create_documents([desc])
        return docs_refine(self.model.llm, docs, SYMPTOM_INITIAL_PROMPT,
                           SYMPTOM_REFINE_PROMPT)

    def __get_cases(self, start_date=None, end_date=None):
        """
        Retrieves cases from Salesforce based on the specified date range.

        Args:
            start_date: The starting date for case retrieval.
            end_date: The ending date for case retrieval.

        Yields:
            Data: An instance of Data containing symptoms, metadata, and
                  case number for each case.
        """
        clause = ''
        conditions = []
        if start_date is not None:
            conditions.append(
                f'LastModifiedDate >= {start_date.isoformat()}T00:00:00Z')

        if end_date is not None:
            conditions.append(
                f'LastModifiedDate < {end_date.isoformat()}T00:00:00Z')

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
                    {'case_number': case['CaseNumber'],
                     'subject': case['Subject']},
                    case['CaseNumber']
                    )

    def get_update_data(self, start_date, end_date):
        """
        Gets updated cases within the specified date range.

        Args:
            start_date: The starting date for case retrieval.
            end_date: The ending date for case retrieval.

        Returns:
            Generator: A generator yielding Data instances for each case.
        """
        return self.__get_cases(start_date, end_date)

    def __translate_into_dialogs(self, records):
        """
        Translates Salesforce case comments into dialog format.

        Args:
            records: A list of case comment records.

        Returns:
            Dialogs: An instance of Dialogs containing translated comments.
        """
        dialogs = Dialogs()
        for comment in records:
            _records = self.sf.query_all(
                        f'SELECT FirstName FROM User WHERE Id = '
                        f'\'{comment["CreatedById"]}\'')
            firstname = _records['records'][0]['FirstName']
            dialogs.append(firstname, comment["CommentBody"])
        return dialogs

    @run_in_parallel(parallelism=4)
    def __condense_context(self, context):
        """
        Condenses the context of case comments into a refined summary.

        Args:
            context: A list of contexts for each user interaction.

        Returns:
            List[str]: A list of condensed summaries for each context.
        """
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=2048,
                chunk_overlap=128,
                length_function=len,
                )
        docs = splitter.create_documents(context)
        return docs_refine(self.model.llm, docs, CONDENSE_INITIAL_PROMPT,
                           CONDENSE_REFINE_PROMPT)

    def __get_process(self, dialogs):
        """
        Generates a summarized process statement from dialog comments.

        Args:
            dialogs: The dialog instances containing user comments.

        Returns:
            str: A summarized process statement based on dialog content.
        """
        user = ''
        contexts = []
        context = []
        for dialog in dialogs:
            if not user or user == dialog['user']:
                user = dialog['user']
                context.append(dialog['comment'])
                continue
            contexts.append((context))
            user = dialog['user']
            context = [dialog['comment']]
        contexts.append((context))
        condensed_contexts = self.__condense_context(contexts)
        process_stmt = ' '.join(condensed_contexts).replace('\n', '')
        return re.sub(r'\s+', ' ', process_stmt).strip()

    @run_in_parallel(parallelism=4)
    def __judge_comment(self, comment):
        """
        Judges the validity of comments to determine their relevance for
        solutions.

        Args:
            comment: A list of comments to be judged.

        Returns:
            List[str | None]: A list of valid comments or None if the comment
                              is judged as not relevant.
        """
        prompt = PromptTemplate.from_template(SOL_JUDGEMENT_PROMPT)
        chain = (
                {'context': RunnablePassthrough()}
                | prompt
                | self.model.llm
                | StrOutputParser()
                )
        result = chain.invoke(comment)
        return comment if result != 'NO' else None

    def __get_solution(self, dialogs):
        """
        Generates a solution summary from the provided dialog comments.

        Args:
            dialogs: The dialog instances containing user comments.

        Returns:
            List[Document]: A list of documents containing refined solution
                            summaries.
        """
        comments = []
        for dialog in dialogs:
            comments.append((dialog['comment']))
        filtered_comments = self.__judge_comment(comments)

        if filtered_comments is None:
            return 'A solution cannot be summarized from the case comments.'
        docs = []
        for comment in filtered_comments:
            if comment is not None:
                docs.append(Document(page_content=comment))
        return docs_refine(self.model.llm, docs, SOL_INITIAL_PROMPT,
                           SOL_REFINE_PROMPT)

    @timed_lru_cache()
    def __get_summary(self, desc, dialogs):
        """
        Creates a summary of the case description and dialogs.

        Args:
            desc: The case description.
            dialogs: The dialog instances containing user comments.

        Returns:
            str: A formatted summary of symptoms, processes, and solutions.
        """
        fn_args = [
            (self.__get_symptom, (desc)),
            (self.__get_process, (dialogs)),
            (self.__get_solution, (dialogs))
            ]
        return '\n'.join(run_fn_in_parallel(fn_args, 3))

    def __get_content(self, case_number):
        """
        Retrieves detailed content for a specified case number.

        Args:
            case_number: The case number for which to retrieve content.

        Returns:
            Content: An instance of Content containing case details and
                     summary.
        """
        case = self.sf.query_all(
                'SELECT Id, Status, Public_Bug_URL__c, Sev_Lvl__c, ' +
                'CaseNumber, Description FROM Case ' +
                f'WHERE CaseNumber = \'{case_number}\'')['records'][0]
        records = self.sf.query_all(f'SELECT CommentBody, '
                                    f'CreatedById FROM CaseComment '
                                    f'WHERE ParentId = \'{case["Id"]}\' '
                                    f'AND IsPublished = True '
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
        """
        Retrieves content for a case based on the provided metadata.

        Args:
            metadata: A dictionary containing case metadata.

        Returns:
            Content: An instance of Content containing case details and
                     summary.
        """
        return self.__get_content(metadata['case_number'])

    def custom_api(self, action, data):
        """
        Handles custom API actions based on specified action type.

        Args:
            action: The action to perform, such as summarizing a case.
            data: The data required for the action.

        Returns:
            Content: The content returned from the specified action.

        Raises:
            ValueError: If the action is not implemented or if required data
                        is missing.
        """
        match action:
            case const.SUMMARIZE_CASE:
                if const.CASE_NUMBER not in data:
                    raise ValueError(
                            f'The {const.CASE_NUMBER} is missing from the '
                            f'data for the {action} action')
                return self.__get_content(data[const.CASE_NUMBER])
            case _:
                raise ValueError(f'The {action} action is not implemented.')

    def generate_output(self, content):
        """
        Generates a formatted output string for the provided case content.

        Args:
            content: An instance of Content containing case metadata and
                     summary.

        Returns:
            str: A formatted string representing the case details, including
                 case number, status, severity level, bug URL, and summary.
        """
        return f'Case:\t\t{content.metadata["case_number"]}\n' \
               f'Status:\t\t{content.metadata["status"]}\n' \
               f'Severity Level:\t{content.metadata["sev_lv"]}\n' \
               f'Bug URL:\t{content.metadata["bug_url"]}\n' \
               f'Summary:\n{content.summary}\n'
