from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class Condition(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the text")

parser = StrOutputParser()

pydantic_parser = PydanticOutputParser(pydantic_object=Condition)

template1 = PromptTemplate(
    template="Analyze the sentiment of the following text and classify it as positive, negative, or neutral: \n {text} \n {format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
)

classification_chain = template1 | model | pydantic_parser

# Create a chain that preserves the original text and adds the sentiment
def add_sentiment(inputs):
    original_text = inputs["text"]["text"]  # Extract the text from nested dict
    sentiment_obj = inputs["sentiment"]
    return {"text": original_text, "sentiment": sentiment_obj.sentiment}

positive_propmt = PromptTemplate(
    template="You are a helpful assistant. You get back to the customer with an appropriate response to the following text: \n {text}",
    input_variables=["text"]
)

negative_propmt = PromptTemplate(
    template=" You are a helpful assistant. You  get back to the customer with an appropriate negative response to the following text: \n {text}",
    input_variables=["text"]
)

neutral_propmt = PromptTemplate(
    template="You are a helpful assistant. You get back to the customer with an appropriate neutral response to the following text: \n {text}",
    input_variables=["text"]
)

positive_chain = positive_propmt | model | parser
negative_chain = negative_propmt | model | parser
neutral_chain = neutral_propmt | model | parser

branch = RunnableBranch(
    (lambda x: x["sentiment"] == 'positive', positive_chain),
    (lambda x: x["sentiment"] == 'negative', negative_chain),
    neutral_chain  # default case
)

# Combine original text with sentiment, then route to appropriate branch
conditional_chain = (
    {"text": RunnablePassthrough(), "sentiment": classification_chain}
    | RunnableLambda(add_sentiment)
    | branch
)

text = '''The phone is terrible. I hate the battery life'''
result = conditional_chain.invoke({"text": text})
print("Result ===> ", result)


