from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def get_gemini_llm(model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    """Initialize and return a Gemini LLM instance.

    Args:
        model (str): The Gemini model to use. Defaults to "gemini-2.5-flash".
    Returns:
        ChatGoogleGenerativeAI: An instance of the Gemini LLM.
    """
    llm = ChatGoogleGenerativeAI(model=model)
    return llm

if __name__ == "__main__":
    gemini_llm = get_gemini_llm()
    response = gemini_llm.invoke("What about today's weather at Dhaka ?")
    # It includes every information of the response
    # print(response)

    # To get only the content of the response
    print(response.content)