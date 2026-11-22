""" sequential chain example """

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from .get_llm import get_llm


def sequential_chains() -> str:
    """ function to execute sequential chain example """

    destination_prompt = PromptTemplate(
        input_variables=["destination"],
        template="I am planning a trip to {destination}. Can you suggest some activities to do there?"
    )

    activities_prompt = PromptTemplate(
        input_variables=["activities"],
        template="I only have one day, so can you create an itinerary from your top three activities: {activities}."
    )

    llm = get_llm()

    seq_chain = (
        {"activities": destination_prompt | llm | StrOutputParser()}
        | activities_prompt
        | llm
        | StrOutputParser()
    )

    return seq_chain.invoke({"destination": "Rome"})
