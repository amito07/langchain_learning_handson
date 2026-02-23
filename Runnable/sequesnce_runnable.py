from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template1 = PromptTemplate(
    template="Write a joke about the following topic.\n {topic}. Select only one.",
    input_variables=["topic"]
)

parser = StrOutputParser()

template2 = PromptTemplate(
    template="Explain the following joke - {joke}",
    input_variables=["joke"]
)

chain = RunnableSequence(template1, model, parser, template2, model, parser)
result = chain.invoke({"topic": "programming"})
print("Result ===> ",result)