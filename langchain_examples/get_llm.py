""" helper function to get llm """

import os

from langchain_openai import ChatOpenAI
from pydantic import SecretStr


LLM_MODEL_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-8b")
LLM_MODEL_API_KEY = os.getenv("LLM_API_KEY", "not-needed")


def get_llm() -> ChatOpenAI:
    """ make and return llm using ChatOpenAI """
    return ChatOpenAI(
        base_url=LLM_MODEL_URL,
        model=LLM_MODEL,
        api_key=SecretStr(LLM_MODEL_API_KEY),
    )
