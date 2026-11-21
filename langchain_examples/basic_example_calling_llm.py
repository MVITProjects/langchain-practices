""" basic example to calling a local llm"""
import os
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


LLM_MODEL_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-8b")
LLM_MODEL_API_KEY = os.getenv("LLM_API_KEY", "not-needed")


def basic_example_calling_llm(prompt: str) -> str:
    """ basic example to call one llm """

    llm = ChatOpenAI(
        base_url=LLM_MODEL_URL,
        model=LLM_MODEL,
        api_key=SecretStr(LLM_MODEL_API_KEY),
    )

    response = llm.invoke(prompt)
    return str(response.content)
