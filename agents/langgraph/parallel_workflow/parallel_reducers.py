from asyncio import selector_events
from asyncio import selector_events
from asyncio import selector_events
from asyncio import selector_events
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

# Create LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature = 0.1,
)

def merge_score_dicts (existing:dict, new_update:dict)-> dict:
    if existing is None:
        return new_update
    return{**existing, **new_update}

# create the state
class AnalyzerState(TypedDict):
    raw_text: str
    safety_scores: Annotated[dict[str,int],merge_score_dicts]

def toxicity_node(state: AnalyzerState) -> dict:
    """Analyze text for toxicity and hate speech."""

    prompt = f"""
    Analyze this text for toxicity and hate speech.

    Give a score from 0-100:
    0 = safe
    100 = extremely toxic or hateful

    Return ONLY the integer.

    Text:
    {state["raw_text"]}
    """

    response = llm.invoke(prompt)

    try:
        # pyrefly: ignore [missing-attribute]
        score = int(response.content.strip())
    except (ValueError, TypeError):
        score = 0

    return {
        "safety_scores": {
            "toxicity_level": score
        }
    }


def copyright_node(state: AnalyzerState) -> dict:
    """Analyze text for plagiarism/copyright similarity."""

    prompt = f"""
    Analyze this text for potential plagiarism or heavy similarity
    to existing content.

    Give a score from 0-100:
    0 = no apparent similarity
    100 = highly likely copied

    Return ONLY the integer.

    Text:
    {state["raw_text"]}
    """

    response = llm.invoke(prompt)

    try:
        # pyrefly: ignore [missing-attribute]
        score = int(response.content.strip())
    except (ValueError, TypeError):
        score = 0

    return {
        "safety_scores": {
            "copyright_level": score
        }
    }


def culture_node(state: AnalyzerState) -> dict:
    """Analyze text for cultural and regional sensitivity."""

    prompt = f"""
    Analyze this text for cultural or regional sensitivity,
    stereotypes, discrimination, or offensive references.

    Give a score from 0-100:
    0 = no sensitivity concerns
    100 = extremely culturally insensitive

    Return ONLY the integer.

    Text:
    {state["raw_text"]}
    """

    response = llm.invoke(prompt)

    try:
        # pyrefly: ignore [missing-attribute]
        score = int(response.content.strip())
    except (ValueError, TypeError):
        score = 0

    return {
        "safety_scores": {
            "cultural_sensitivity_level": score
        }
    }

# Create graph
# pyrefly: ignore [bad-specialization]
builder = StateGraph(AnalyzerState)

builder.add_node("toxicity_node", toxicity_node)
builder.add_node("copyright_node", copyright_node)
builder.add_node("culture_node", culture_node)

builder.add_edge(START, "toxicity_node")
builder.add_edge(START, "copyright_node")
builder.add_edge(START, "culture_node")

builder.add_edge("toxicity_node", END)
builder.add_edge("copyright_node", END)
builder.add_edge("culture_node", END)

app = builder.compile()

sample_text = """
    Yo guys! Welcome back to the stream. Today I am going to show you how to hack into
    your friend's system using a script I copied directly from an online forum.
    Honestly, traditional security protocols are absolute garbage and anyone still using
    them is an absolute idiot. Let's dive into the code!    
"""

final_state = app.invoke({
    "raw_text": sample_text,
    "safety_scores": {}
})

print(final_state["safety_scores"])