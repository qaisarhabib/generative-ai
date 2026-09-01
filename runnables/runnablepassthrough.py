from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2506"
)

parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
        ("system", "you are a code generator"),
        ("human", "Explain {topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
        ("system", "you are helpful assistant who can explain the code"),
        ("human", "Explain the following code in simple words:\n{code}")
])

seq = code_prompt | model | parser | explain_prompt | model | parser

seq. invoke