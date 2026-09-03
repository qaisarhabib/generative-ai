from langchain.agents.middleware.types import wrap_tool_call
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain.agents import create_agent


from tavily import TavilyClient
from rich import print


# ============================================================
# CONFIG
# ============================================================

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ============================================================
# WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""

    if not OPENWEATHER_API_KEY:
        return "Weather service is not configured. OPENWEATHER_API_KEY is missing."

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
        )

        data = response.json()

    except requests.RequestException as e:
        return f"Weather service error: {str(e)}"

    if response.status_code != 200:
        return f"Could not get weather for {city}: {data.get('message', 'Unknown error')}"

    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]

    description = data["weather"][0]["description"]

    return (
        f"Weather in {data['name']}: "
        f"{description}, "
        f"{temp}°C, "
        f"feels like {feels_like}°C, "
        f"humidity {humidity}%."
    )


# ============================================================
# TAVILY CLIENT
# ============================================================

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


# ============================================================
# NEWS TOOL
# ============================================================

@tool
def get_news(city: str) -> str:
    """Get the latest news about a city."""

    if not TAVILY_API_KEY:
        return "News service is not configured. TAVILY_API_KEY is missing."

    try:
        response = tavily_client.search(
            query=f"latest news about {city}",
            topic="news",
            search_depth="basic",
            max_results=3,
        )

    except Exception as e:
        return f"News service error: {str(e)}"

    results = response.get("results", [])

    if not results:
        return f"No recent news found for {city}."

    news = []

    for result in results:
        title = result.get("title", "No title")
        content = result.get("content", "No description")
        url = result.get("url", "")

        news.append(
            f"Title: {title}\n"
            f"Summary: {content}\n"
            f"Source: {url}"
        )

    return "\n\n".join(news)


@wrap_tool_call
def human_approval(request, handler):
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]

    print("\n")
    print(f"Agent wants to call: {tool_name}")
    print(f"Arguments: {tool_args}")

    approval = input("Approve tool call? (yes/no): ").strip().lower()

    if approval in ["yes", "y"]:
        print(f"Executing {tool_name}...\n")

        return handler(request)

    print(f"Tool call {tool_name} rejected.\n")

    return ToolMessage(
        content=f"The user denied permission to call {tool_name}.",
        tool_call_id=request.tool_call["id"],
    )

# ============================================================
# LLM
# ============================================================

llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0,
)

agent = create_agent(
    llm, 
    tools= [get_weather, get_news],
    system_prompt="""
        You are a City Intelligence Assistant.

        You can help users with:
        1. Current weather using get_weather.
        2. Latest city news using get_news.

        Rules:
        - Use get_weather when the user asks about weather.
        - Use get_news when the user asks about recent or latest news.
        - You may use both tools if the user asks for both weather and news.
        - Do not call tools for normal conversation or greetings.
        - Answer clearly and concisely.
        - Never invent weather or news information.
    """,
    middleware=[human_approval],
)

print("[bold cyan]City Intelligence System[/bold cyan]")
print("[dim]Type 'exit' to quit.[/dim]\n")

messages = []

while True:

    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        result = agent.invoke({
            "messages": messages
        })

        messages = result["messages"]

        print("AI:", messages[-1].content)

    except Exception as e:
        print(f"[red]Error:[/red] {e}")