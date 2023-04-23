from langchain.chains.question_answering import load_qa_chain

class QAChain:
    def __init__(self, chain_type, llm, vectordb, prompt_generator):
        prompt = prompt_generator.get_prompt()

        if prompt is not None:
            self.qa_chain = load_qa_chain(llm=llm.llm, chain_type=chain_type, prompt=prompt)
        else:
            self.qa_chain = load_qa_chain(llm=llm.llm, chain_type=chain_type)
        self.vectordb = vectordb

    def ask(self, question):
        docs = self.vectordb.similarity_search(question)
        return self.qa_chain({"input_documents": docs, "question": question}, return_only_outputs=True)
