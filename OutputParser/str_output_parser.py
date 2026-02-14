from langchain_core.output_parsers import StrOutputParser
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
    groq_model = get_groq_llm()

    template1 = PromptTemplate(template= "Summarize the following {topic} topic", input_variables=["topic"] )
    template2 = PromptTemplate(template= "Write down 5 lines summary on the following {text} text", input_variables=["text"] )

    parse = StrOutputParser()

    chain = template1 | groq_model | parse | template2 | groq_model | parse

    response = chain.invoke({'topic': 'Black Holes'})

    print(response)