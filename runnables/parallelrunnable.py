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

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

detail_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

chain = RunnableParallel({
    "short": (
        RunnableLambda(lambda x: {"topic": x["short"]})
        | short_prompt
        | model
        | parser
    ),

    "detail": (
        RunnableLambda(lambda x: {"topic": x["detail"]})
        | detail_prompt
        | model
        | parser
    )
})

result = chain.invoke({
    "short": "Machine learning",
    "detail": "Deep learning"
})

print("SHORT:")
print(result["short"])

print("\nDETAIL:")
print(result["detail"])