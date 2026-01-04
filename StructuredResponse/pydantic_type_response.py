from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal, Optional

load_dotenv()

class Schema(BaseModel):
    key_themes: list[str] = Field(description="Write down the key themes discussed in the text.")
    summary: str = Field(description="Provide a concise brief summary of the text.")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Determine the overall sentiment of the text.")
    pros: Optional[list[str]] = Field(default=None, description="List the positive aspects mentioned in the text.")
    cons: Optional[list[str]] = Field(default=None, description="List the negative aspects mentioned in the text.")
    name: Optional[str] = Field(default=None, description="Write down the name of the product or service mentioned in the text.")


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

    text = (
        "Recently, I purchased the Acme SuperWidget 3000, and I must say, it has exceeded my expectations in many ways. If you're looking for a reliable and efficient widget, this is the one to get. Lets start with the pros: the build quality is exceptional, and it performs tasks quickly and accurately. The user interface is intuitive, making it easy for anyone to use. Additionally, the customer support from Acme has been top-notch, responding promptly to my inquiries."
    )

    structured_model = groq_llm.with_structured_output(Schema, strict=True)
    response = structured_model.invoke(text)
    print(response)