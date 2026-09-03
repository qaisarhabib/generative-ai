from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

search_tool = TavilySearchResults(max_result = 5)

llm = ChatMistralAI(
    model="mistral-small-2506"
)

prompt = ChatPromptTemplate.from_template(
    """
    you are a helpful assistant
    summarize the following news into clear bullet points
    {news}
    """
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.run("Latest war news in 2026")

result = chain.invoke({"news": news_result})

print(result)