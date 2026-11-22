""" example of chat models """
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


LLM_MODEL_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-8b")
LLM_MODEL_API_KEY = os.getenv("LLM_API_KEY", "not-needed")


def chat_models_example() -> str:
    """ def to execute chat models example """
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a calculator that responds with math."),
        ("human", "Answer this math question: What is two plus two?"),
        ("ai", "2+2=4"),
        ("human", "Answer this math question: {math}")
    ])

    llm = ChatOpenAI(
        base_url=LLM_MODEL_URL,
        model=LLM_MODEL,
        api_key=SecretStr(LLM_MODEL_API_KEY),
    )

    llm_chain = template | llm

    response = llm_chain.invoke({"math": "What is five times five?"})

    return str(response.content)
