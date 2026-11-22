""" example of chat models """

from langchain_core.prompts import ChatPromptTemplate

from .get_llm import get_llm


def chat_models_example() -> str:
    """ def to execute chat models example """
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a calculator that responds with math."),
        ("human", "Answer this math question: What is two plus two?"),
        ("ai", "2+2=4"),
        ("human", "Answer this math question: {math}")
    ])

    llm = get_llm()

    llm_chain = template | llm

    response = llm_chain.invoke({"math": "What is five times five?"})

    return str(response.content)
