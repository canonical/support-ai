from operator import itemgetter
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from lib.const import CONFIG_BASIC_MODEL
from lib.datasources.ds_querier import DSQuerier
from lib.model_manager import ModelManager


class Chain:
    def __init__(self, config):
        if CONFIG_BASIC_MODEL not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_BASIC_MODEL}')
        self.basic_model = ModelManager(config[CONFIG_BASIC_MODEL])
        self.ds_querier = DSQuerier(config)
        self.memories = {}

    def __get_memory(self, session):
        if session not in self.memories:
            self.memories[session] = ConversationSummaryBufferMemory(llm=self.basic_model.llm,
                                                                     max_token_limit=120,
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

    def ask(self, query, ds_type=None, session=None):
        ds, doc = self.ds_querier.query(query, ds_type)
        content = ds.get_content(doc.metadata)
        if session is not None:
            memory = self.__get_memory(session)
            content.Summary = self.__get_summary_with_memory(memory, query, content.Summary)
            memory.save_context({'input': query}, {'output': content.Summary})
        return ds.generate_output(content)
