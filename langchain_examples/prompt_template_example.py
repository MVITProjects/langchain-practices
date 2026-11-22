""" example using a PromptTemplate with a local llm"""

from langchain_core.prompts import PromptTemplate

from .get_llm import get_llm


def prompt_template_example(concept: str) -> str:
    """ explain a concept using a PromptTemplate piped into an llm """

    template = "Expain this concept simply and concisely: {concept}"

    prompt_template = PromptTemplate.from_template(template=template)

    llm = get_llm()

    llm_chain = prompt_template | llm

    response = llm_chain.invoke({"concept": concept})
    return str(response.content)
