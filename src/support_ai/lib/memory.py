"""
This module provides a memory management class that stores and retrieves
conversation history in a MongoDB database.
"""

from operator import itemgetter
from threading import Lock

from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import (
    MongoDBChatMessageHistory
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from . import const


class Memory:
    """
    Manages session-based conversational memory for a chatbot, allowing
    integration of a history-based context using MongoDB storage and
    Langchain's LLM framework.
    """

    def __init__(self, config, llm):
        """
        Initializes the Memory class with a database connection and a
        language model.

        Args:
            config: Configuration dictionary containing the MongoDB connection
                    string.
            llm: Language model to handle query processing.

        Raises:
            ValueError: If the configuration does not contain a database
                        connection string.
        """
        if const.CONFIG_DB_CONNECTION not in config:
            raise ValueError(
                f'The config doesn\'t contain {const.CONFIG_DB_CONNECTION}')
        self.db_connection = config[const.CONFIG_DB_CONNECTION]
        self.llm = llm
        self.mutex = Lock()
        self.session_memories = {}

    def __get_session_memory(self, session):
        """
        Retrieves or creates a memory object for a given session.

        Args:
            session: Session ID to retrieve or create memory for.

        Returns:
            ConversationSummaryBufferMemory: Memory object associated
                                             with the session.
        """
        with self.mutex:
            if session not in self.session_memories:
                memory = MongoDBChatMessageHistory(
                        connection_string=self.db_connection,
                        session_id=session
                        )
                self.session_memories[session] = \
                    ConversationSummaryBufferMemory(chat_memory=memory,
                                                    llm=self.llm,
                                                    return_messages=True)
            return self.session_memories[session]

    def integrate(self, session, query, context):
        """
        Integrates a query with the current session memory, generating
        a response based on past interactions and provided context.

        Args:
            session: Session ID for memory storage.
            query: The user query to be integrated with context.
            context: Additional contextual information for the query.

        Returns:
            str: The generated response based on query and context.
        """
        memory = self.__get_session_memory(session)
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are a helpful chatbot'),
            MessagesPlaceholder(variable_name='history'),
            ('human', 'Based on the context: {context}, {query}'),
            ])
        chain = (
                RunnablePassthrough.assign(
                    history=RunnableLambda(memory.load_memory_variables) |
                    itemgetter('history')
                    )
                | prompt
                | self.llm
                | StrOutputParser()
                )
        integrated_context = chain.invoke({'context': context, 'query': query})
        memory.save_context({'input': query}, {'output': integrated_context})
        return integrated_context

    def clear(self, session):
        """
        Clears the conversation memory for a given session.

        Args:
            session: Session ID for which to clear memory.
        """
        with self.mutex:
            memory = MongoDBChatMessageHistory(
                    connection_string=self.db_connection,
                    session_id=session
                    )
            ConversationSummaryBufferMemory(chat_memory=memory).clear()
            if session in self.session_memories:
                del self.session_memories[session]
