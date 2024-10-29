"""
This module provides functionality for interacting with a Salesforce
Knowledge Base,
"""
from html.parser import HTMLParser
from io import StringIO

import simple_salesforce
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from support_ai.lib import const
from support_ai.lib.context import BaseContext
from support_ai.lib.utils.lru import timed_lru_cache
from support_ai.lib.datasources.ds import Data, Content, Datasource


QUESTIONS_PROMPT = """
Generate five questions that can be answered by the article with the summary:
    "{summary}"
    QUESTIONS:""".strip()
SUMMARY_PROMPT = """Write a concise summary of the following:
    "{solution}"
    CONCISE SUMMARY:"""


class HtmlTagStripper(HTMLParser):
    """
    A custom HTML parser that strips HTML tags and retains the text content.
    """

    def __init__(self):
        """
        Initialize the HtmlTagStripper and its attributes.
        """
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, data):
        """
        Handle the text data found between HTML tags.

        Args:
            data: The text data to be retained.
        """
        self.text.write(data)

    def get_data(self):
        """
        Retrieve the accumulated text content without HTML tags.

        Returns:
            str: The text content extracted from the HTML input.
        """
        return self.text.getvalue()


def strip_tags(html):
    """
    Removes HTML tags from the given string.

    Args:
        html: The input HTML string to strip.

    Returns:
        str: The text content extracted from the HTML, without any tags.
    """
    stripper = HtmlTagStripper()
    stripper.feed(html)
    return stripper.get_data()


def get_authentication(auth_config):
    """
    Validates and retrieves authentication details from the provided
    configuration.

    Args:
        auth_config: The authentication configuration containing
                     username, password, and token.

    Raises:
        ValueError: If any of the required authentication fields are missing.

    Returns:
        dict: A dictionary containing the validated username, password,
              and token.
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


class KnowledgeBaseSource(BaseContext, Datasource):
    """
    A class for retrieving and processing articles from a Salesforce
    Knowledge Base.
    """

    def __init__(self, config):
        super().__init__(config)
        if const.CONFIG_AUTHENTICATION not in config:
            raise ValueError(
                f'The config doesn\'t contain {const.CONFIG_AUTHENTICATION}')
        auth = get_authentication(config[const.CONFIG_AUTHENTICATION])
        self.sf = simple_salesforce.Salesforce(**auth)
        self.model = self.model_manager.get_model(config)

    def __generate_qeustions(self, summary):
        """
        Generates questions based on the provided article summary.

        Args:
            summary: The summary of the article for which to generate
                     questions.

        Returns:
            list: A list of questions generated from the summary.
        """
        prompt = PromptTemplate.from_template(QUESTIONS_PROMPT)
        chain = (
                {'summary': RunnablePassthrough()}
                | prompt
                | self.model.llm
                | StrOutputParser()
                )
        return chain.invode(summary)

    def __get_articles(self, start_date=None, end_date=None):
        """
        Retrieves articles from the Salesforce Knowledge Base within the
        specified date range.

        Args:
            start_date: The start date for filtering articles.
            end_date: The end date for filtering articles.

        Yields:
            Data: A Data object containing generated questions, metadata,
                  and article ID.
        """
        clause = ''
        conditions = []
        if start_date is not None:
            conditions.append(
                f'LastModifiedDate >= {start_date.isoformat()}T00:00:00Z')
        if end_date is not None:
            conditions.append(
                f'LastModifiedDate < {end_date.isoformat()}T00:00:00Z')
        conditions.append(
            'Knowledge_1_Approval_Status__c = \'Approval Complete\'')
        conditions.append(
            'PublishStatus = \'Online\'')

        for condition in conditions:
            if clause:
                clause += ' AND '
            clause += condition
        sql_cmd = 'SELECT Id, KnowledgeArticleId, Title, Summary ' + \
                  'FROM Knowledge__kav' + \
                  (f' WHERE {clause}' if clause else '')
        articles = self.sf.query_all(sql_cmd)

        for article in articles['records']:
            yield Data(
                    self.__generate_qeustions(article['Summary']),
                    {'article_id': article['KnowledgeArticleId'],
                     'title': article['Title']},
                    article['Id']
            )

    def get_update_data(self, start_date, end_date):
        """
        Retrieves update data for the specified date range.

        Args:
            start_date: The start date for filtering articles.
            end_date: The end date for filtering articles.

        Returns:
            generator: A generator yielding Data objects for each article.
        """
        return self.__get_articles(start_date, end_date)

    @timed_lru_cache()
    def __get_summary(self, solution):
        """
        Generates a concise summary for the provided solution.

        Args:
            solution: The solution text to summarize.

        Returns:
            str: The concise summary generated for the solution.
        """
        prompt = PromptTemplate.from_template(SUMMARY_PROMPT)
        chain = (
                {'solution': RunnablePassthrough()}
                | prompt
                | self.model.llm
                | StrOutputParser()
                )
        return chain.invoke(solution)

    def get_content(self, metadata):
        """
        Retrieves the content of an article based on its metadata.

        Args:
            metadata: The metadata containing the article ID.

        Returns:
            Content: A Content object containing the summary of the article.
        """
        article = self.sf.query_all(
            f'SELECT Knowledge_1_Solution__c FROM Knowledge__kav '
            f'WHERE KnowledgeArticleId = \'{metadata["article_id"]}\'')
        return Content(
                {},
                self.__get_summary(
                    strip_tags(article['records'][0]
                               ['Knowledge_1_Solution__c']))
                )

    def custom_api(self, action, data):
        """
        Custom API action placeholder.

        Args:
            action: The action to perform.
            data: The data required for the action.

        Raises:
            ValueError: Indicates that the action is not implemented.
        """
        raise ValueError(f'The {action} action is not implemented.')

    def generate_output(self, content):
        """
        Generates the output for the provided content.

        Args:
            content: The content object to generate output from.

        Returns:
            str: The summary of the content.
        """
        return content.summary
