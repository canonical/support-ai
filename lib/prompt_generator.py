from typing import Optional
from langchain.prompts import PromptTemplate

class PromptGenerator:
    def get_prompt(self) -> Optional[PromptTemplate]:
        template = """
        Please select three important things that one must know for the problem.
        QUESTION: {question}
        =========
        {context}
        =========
        FINAL ANSWER:"""
        return PromptTemplate(input_variables=['context', 'question'], template=template)
