""" basic example to calling a local llm"""
from .get_llm import get_llm


def basic_example_calling_llm(prompt: str) -> str:
    """ basic example to call one llm """

    llm = get_llm()

    response = llm.invoke(prompt)
    return str(response.content)
