from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

chatModel = ChatGroq(model="llama-3.1-8b-instant")

template1 = PromptTemplate(
    template="Write a detailed summary of the following topics: {topics}",
    input_variables=["topics"]
)

template2 = PromptTemplate(
    template="Figure out top 5 keywords from the following summary: {summary}",
    input_variables=["summary"]
)

parser = StrOutputParser()

chain = template1 | chatModel | parser | template2 | chatModel | parser

result = chain.invoke({"topics": "Artificial Intelligence, Machine Learning, Deep Learning"})

print("Result ===> ",result)