from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1 = PromptTemplate(
    template="Write a Detailed report on the following topic: {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {topic}.",
    input_variables=["topic"]
)

parser = StrOutputParser()

res_gen_result = RunnableSequence(prompt1, model, parser)

branch = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(res_gen_result, branch)

print(final_chain.invoke({"topic": "Bangldesh 1971 history"}))