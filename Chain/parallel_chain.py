from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGroq(model="llama-3.1-8b-instant")
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

template1 = PromptTemplate(
    template="Write down short and simple notes based on the following topics: \n {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Generate 5 short questions based on the following topic:  \n {topic}",
    input_variables=["topic"]
)

template3 = PromptTemplate(
    template="Merge the provided notes and questions into a single document. \n Notes: {notes} \n Questions: {questions}",
    input_variables=["notes", "questions"]
)

parallel_chain = RunnableParallel({
    "notes": template1 | model1 | parser,
    "questions": template2 | model2 | parser
})

merge_chain = template3 | model1 | parser

chain = parallel_chain | merge_chain

text = ''' Cross decomposition algorithms find the fundamental relations between two matrices (X and Y). They are latent variable approaches to modeling the covariance structures in these two spaces. They will try to find the multidimensional direction in the X space that explains the maximum multidimensional variance direction in the Y space. In other words, PLS projects both X and Y into a lower-dimensional subspace such that the covariance between transformed(X) and transformed(Y) is maximal.

PLS draws similarities with Principal Component Regression (PCR), where the samples are first projected into a lower-dimensional subspace, and the targets y are predicted using transformed(X). One issue with PCR is that the dimensionality reduction is unsupervised, and may lose some important variables: PCR would keep the features with the most variance, but it’s possible that features with small variances are relevant for predicting the target. In a way, PLS allows for the same kind of dimensionality reduction, but by taking into account the targets y. An illustration of this fact is given in the following example: * Principal Component Regression vs Partial Least Squares Regression.

Apart from CCA, the PLS estimators are particularly suited when the matrix of predictors has more variables than observations, and when there is multicollinearity among the features. By contrast, standard linear regression would fail in these cases unless it is regularized.'''

result = chain.invoke({"topic": text})

print("Result ===> ",result)


