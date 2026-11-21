""" example using a PromptTemplate with a local llm"""
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


LLM_MODEL_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-8b")
LLM_MODEL_API_KEY = os.getenv("LLM_API_KEY", "not-needed")


def prompt_template_example(concept: str) -> str:
    """ explain a concept using a PromptTemplate piped into an llm """

    template = "Expain this concept simply and concisely: {concept}"

    prompt_template = PromptTemplate.from_template(template=template)

    llm = ChatOpenAI(
        base_url=LLM_MODEL_URL,
        model=LLM_MODEL,
        api_key=SecretStr(LLM_MODEL_API_KEY),
    )

    llm_chain = prompt_template | llm

    response = llm_chain.invoke({"concept": concept})
    return str(response.content)
