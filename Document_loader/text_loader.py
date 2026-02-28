from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()
template1 = PromptTemplate(
    template="Summarize the following poem in a few sentences: \n {poem}",
    input_variables=["poem"]
)

loader = TextLoader("ai_poem.txt")

documents = loader.load()

chain = template1 | model | parser

result = chain.invoke({"poem": documents[0].page_content})
print("Result ===> ",result)