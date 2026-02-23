from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()
model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

template1 = PromptTemplate(
    template="Write a short tweet about the following topic.\n {topic}. Pick One",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a short LinkedIn post about the following topic.\n {topic}. Pick One",
    input_variables=["topic"]
)

parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(template1, model1, parser),
    "linkedin_post": RunnableSequence(template2, model1, parser)
})

result = parallel_chain.invoke({"topic": "the benefits of using AI in business"})
print("Result ===> ",result)

print("Tweet ===> ",result["tweet"])
print("LinkedIn Post ===> ",result["linkedin_post"])