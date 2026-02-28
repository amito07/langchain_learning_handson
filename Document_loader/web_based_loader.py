from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

propmt1 = PromptTemplate(
    template="Summarize the product information from the following webpage: \n {webpage}",
    input_variables=["webpage"]
)

loader = WebBaseLoader("https://www.applegadgetsbd.com/product/macbook-air-m4-15-inch")
documents = loader.load()

chain = propmt1 | model | parser
result = chain.invoke({"webpage": documents[0].page_content})
print("Result ===> ",result)