""" example of few shot prompts """
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

from .get_llm import get_llm


def few_shot_prompts_example() -> str:
    """ function to execute few shot prompts example"""

    examples = [
        {
            "question": "Does Henry Campbell have any pets?",
            "answer": "Henry Campbell has a dog called Pluto."
        }
    ]

    # ! Convert pandas DataFrame to list of dicts
    # examples = df.to_dict(orient="records")

    example_prompt = PromptTemplate.from_template(
        "Question: {question}\n{answer}")

    prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"]
    )

    llm = get_llm()

    llm_chain = prompt_template | llm

    response = llm_chain.invoke(
        {"input": "What is the name of Henry Campbell's dog?"})

    return str(response.content)
