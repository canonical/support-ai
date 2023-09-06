"""Generate prompt for question answering chain"""""
from typing import Optional
from langchain.prompts import PromptTemplate
from const import CONFIG_PROMPT, CONFIG_SETTING

THREE_THINGS_PROMPT = """
Please select three important things that one must know for the problem.
QUESTION: {question}
=========
{context}
=========
FINAL ANSWER:"""


class PromptGenerator:
    """Generate prompt for question answering chain"""

    def __init__(self, config) -> None:
        """Initialize prompt generator"""
        if CONFIG_PROMPT not in config[CONFIG_SETTING]:
            raise ValueError(f'The configuration\'s {CONFIG_SETTING} section doesn\'t contain {CONFIG_PROMPT}')
        self.tempalte_type = config[CONFIG_SETTING][CONFIG_PROMPT]

    def get_prompt(self) -> Optional[PromptTemplate]:
        """Get prompt template for question answering chain

        Args:
            config: Configuration for prompt generator

        Returns:
            Prompt template for question answering chain
        """

        prompt_template = None
        prompt_type = self.tempalte_type
        template = None

        match prompt_type:
            # In default case, we don't need to specify the prompt_template
            case 'default':
                pass
            case 'three_things':
                template = THREE_THINGS_PROMPT
            case _:
                print(
                    f"Unknown prompt type [{prompt_type}] and will use "
                    f"default prompt from langchain."
                )

        if template is not None:
            prompt_template = PromptTemplate(
                input_variables=['context', 'question'],
                template=template
            )

        return prompt_template
