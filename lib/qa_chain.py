from langchain.chains.question_answering import load_qa_chain

class QAChain:
    def __init__(self, chain_type, max_res_num, llm, vectordb, prompt_generator):
        prompt = prompt_generator.get_prompt()

        if prompt is not None:
            self.qa_chain = load_qa_chain(llm=llm.llm, chain_type=chain_type, prompt=prompt)
        else:
            self.qa_chain = load_qa_chain(llm=llm.llm, chain_type=chain_type)
        self.vectordb = vectordb
        self.max_res_num = max_res_num

    def ask(self, question):
        replies = []
        docs_and_scores = self.vectordb.db.similarity_search_with_score(question, k=self.max_res_num)
        docs_and_scores.sort(key=lambda doc_and_score: doc_and_score[1], reverse=True)
        for doc, _ in docs_and_scores:
            result = self.qa_chain({"input_documents": [doc], "question": question}, return_only_outputs=True)
            replies.append(result['output_text'])
        return replies
