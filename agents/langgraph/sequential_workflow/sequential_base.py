from typing import TypedDict

# create the state
class PipelineState(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Create LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
)

# Editor node
def editor_node(state: PipelineState) -> dict:
    """Stage 1: Clean grammar, remove typos, and refine the tone."""

    prompt = f"""
    You are a professional editor.

    Edit the following text:
    - Fix grammar mistakes
    - Remove spelling errors and typos
    - Improve sentence structure
    - Make the tone clear and natural
    - Preserve the original meaning
    - Do not add unnecessary information

    Text:
    {state["raw_input"]}

    Return only the edited text.
    """

    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        edited_text = response.content.strip()
    else:
        edited_text = "".join(
            block["text"] if isinstance(block, dict) else str(block)
            for block in response.content
        ).strip()

    return {
        "edited_text": edited_text
    }

def script_writer_node(state: PipelineState) -> dict:
    """Stage 2: Convert the edited text into a clear and engaging video script."""

    prompt = f"""
    You are a professional video script writer.

    Convert the following edited text into a clear and engaging video script.

    Requirements:
    - Create a strong opening hook.
    - Keep the script easy to understand.
    - Use natural, conversational language.
    - Organize the content with a clear beginning, middle, and ending.
    - Preserve the important information from the original text.
    - Do not invent facts or information.
    - Make it suitable for a short educational video.
    - Return only the script.

    Edited text:
    {state["edited_text"]}
    """

    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        script_text = response.content.strip()
    else:
        script_text = "".join(
            block["text"] if isinstance(block, dict) else str(block)
            for block in response.content
        ).strip()

    return {
        "script_text": script_text
    }

def translator_node(state: PipelineState) -> dict:
    """Stage 3: Translate the script into natural Roman Urdu."""

    prompt = f"""
    You are a professional Roman Urdu translator.

    Translate the following video script into natural, conversational Roman Urdu.

    Requirements:
    - Use Roman Urdu written with English letters.
    - Keep the language natural and easy to speak.
    - Do not translate word-for-word if it sounds unnatural.
    - Preserve the original meaning and important information.
    - Keep technical terms such as AI, Python, API, and LangGraph in English when appropriate.
    - Do not add new information.
    - Do not use Urdu/Arabic script.
    - Return only the translated script.

    Script:
    {state["script_text"]}
    """

    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        translated_text = response.content.strip()
    else:
        translated_text = "".join(
            block["text"] if isinstance(block, dict) else str(block)
            for block in response.content
        ).strip()

    return {
        "final_output": translated_text
    }

from langgraph.graph import StateGraph, START, END

# Create graph
# pyrefly: ignore [bad-specialization]
graph = StateGraph(PipelineState)

graph.add_node("editor", editor_node)
graph.add_node("script_writer", script_writer_node)
graph.add_node("translator", translator_node)

# Define workflow
graph.add_edge(START, "editor")
graph.add_edge("editor", "script_writer")
graph.add_edge("script_writer", "translator")
graph.add_edge("translator", END)


# Compile graph
app = graph.compile()

result = app.invoke({
    "raw_input": "Ai gents are the future of tech, they can think plan ac act on",})

print(result["final_output"])