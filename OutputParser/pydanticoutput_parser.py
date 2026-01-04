from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()

def get_groq_llm(model: str = "llama-3.1-8b-instant") -> ChatGroq:
    """Initialize and return a Groq LLM instance.

    Args:
        model (str): The Groq model to use. Defaults to "llama-3.1-8b-instant".
    Returns:
        ChatGroq: An instance of the Groq LLM.
    """
    llm = ChatGroq(model=model)
    return llm

class Person(BaseModel):
    name: str = Field(description="The name of the person.")
    age: int = Field(description="The age of the person.")
    city: str = Field(description="The city where the person lives.")

if __name__ == "__main__":
    groq_llm = get_groq_llm()

    parser = PydanticOutputParser(pydantic_object=Person)

    template = PromptTemplate(template="Give me the name, age and city of a fictional character who lives in {country}\n\n. Return the response as following format {format_instructions}\n\n",
          input_variables=["country"],
          partial_variables={"format_instructions": parser.get_format_instructions()}) 

    chain = template | groq_llm | parser

    response = chain.invoke({"country": "India"})
    print(response)
