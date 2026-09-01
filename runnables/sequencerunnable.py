from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Prompt template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# Model
model = ChatMistralAI(
    model="mistral-small-2506"
)

# Output Parser
parser = StrOutputParser()

runnable = prompt | model | parser

topic = "machine learning"

result = runnable.invoke({"topic":topic})

print(result)
