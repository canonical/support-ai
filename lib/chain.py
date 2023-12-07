from operator import itemgetter
from threading import Lock
from langchain.memory import ConversationSummaryBufferMemory, MongoDBChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from lib.const import CONFIG_BASIC_MODEL, CONFIG_CHAT_MEMORY, CONFIG_DB_CONNECTION
from lib.datasources.ds_querier import DSQuerier
from lib.model_manager import ModelManager


class Chain:
    def __init__(self, config):
        if CONFIG_BASIC_MODEL not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_BASIC_MODEL}')
        self.basic_model = ModelManager(config[CONFIG_BASIC_MODEL])
        if CONFIG_CHAT_MEMORY not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_CHAT_MEMORY}')
        if CONFIG_DB_CONNECTION not in config[CONFIG_CHAT_MEMORY]:
            raise ValueError(f'The {CONFIG_CHAT_MEMORY} doesn\'t contain {CONFIG_DB_CONNECTION}')
        self.db_connection = config[CONFIG_CHAT_MEMORY][CONFIG_DB_CONNECTION]
        self.ds_querier = DSQuerier(config)
        self.mutex = Lock()
        self.memories = {}

    def __get_memory(self, session):
        with self.mutex:
            if session not in self.memories:
                chat_memory = MongoDBChatMessageHistory(
                        connection_string=self.db_connection,
                        session_id=session
                        )
                self.memories[session] = ConversationSummaryBufferMemory(chat_memory=chat_memory,
                                                                         llm=self.basic_model.llm,
                                                                         return_messages=True
                                                                         )
            return self.memories[session]

    def __get_summary_with_memory(self, memory, query, context):
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are a helpful chatbot'),
            MessagesPlaceholder(variable_name='history'),
            ('human', 'Based on the context: {context}, {query}'),
            ])
        chain = (
                RunnablePassthrough.assign(
                    history=RunnableLambda(memory.load_memory_variables) | itemgetter('history')
                    )
                | prompt
                | self.basic_model.llm
                | StrOutputParser()
                )
        return chain.invoke({'context': context, 'query': query})

    def __stream(self, output):
        delimiters = [' ', '\t', '\n']
        l, r = 0, 0
        for r in range(len(output)):
            if output[r] in delimiters:
                yield output[l:r+1]
                l = r + 1
        if l < r:
            yield output[l:r+1]

    def ask(self, query, ds_type=None, session=None):
        ds, doc = self.ds_querier.query(query, ds_type)
        content = ds.get_content(doc.metadata)
        if session is not None:
            memory = self.__get_memory(session)
            content.Summary = self.__get_summary_with_memory(memory, query, content.Summary)
            memory.save_context({'input': query}, {'output': content.Summary})
        return self.__stream(ds.generate_output(content))
