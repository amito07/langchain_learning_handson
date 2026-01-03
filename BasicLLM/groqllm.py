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
    response = groq_llm.invoke("Tell me about Dhaka in short summary?")
    # It includes every information of the response
    # print(response)

    # To get only the content of the response
    print(response.content)