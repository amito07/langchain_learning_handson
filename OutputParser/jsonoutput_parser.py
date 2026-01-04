from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

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

if __name__ == "__main__":
    groq_llm = get_groq_llm()

    parser = JsonOutputParser()

    template = PromptTemplate(template="Give me the name, age and city of a fictional character who lives in Bangladesh. {format_instructions}",
          input_variables=[],
          partial_variables={"format_instructions": parser.get_format_instructions()}) 

    chain = template | groq_llm | parser

    response = chain.invoke({})
    print(response)                  