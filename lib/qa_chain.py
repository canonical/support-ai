from langchain.chains import RetrievalQA

class QAChain:
    def __init__(self, chain_type, llm, vectordb, prompt_generator):
        prompt = prompt_generator.get_prompt()

        if prompt is not None:
            self.qa_chain = RetrievalQA.from_chain_type(llm=llm.llm, chain_type=chain_type,
                                                        chain_type_kwargs={'prompt': prompt}, retriever=vectordb.db.as_retriever())
        else:
            self.qa_chain = RetrievalQA.from_chain_type(llm=llm.llm, chain_type=chain_type, retriever=vectordb.db.as_retriever())

    def ask(self, question):
        return self.qa_chain.run(question)
