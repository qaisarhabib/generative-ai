from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool

from rich import print

load_dotenv()

# creating a tool

@tool
def get_text_length(text:str) -> int:
    """ Return the number of character in a given text """
    return len(text)


tools={
    "get_text_length" : get_text_length
}
llm = ChatMistralAI(
    model="mistral-small-2506"
)

#tool binding
llm_with_tool = llm.bind_tools([get_text_length])

message= []
prompt = input("You: ")
query= HumanMessage(prompt)

message.append(query)

result = llm_with_tool.invoke(message)

message.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message =tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)

result = llm_with_tool.invoke(message)
print(result.content)