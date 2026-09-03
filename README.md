# Generative AI Engineering Workspace

Welcome to the **Generative AI Engineering Workspace**! This repository is organized modularly, with `generative-ai/` as the root workspace and each AI architecture, system component, or agent pipeline contained within its own dedicated module.

---

## 📁 Repository Structure

```text
generative-ai/
├── agents/                   # AI Agent architectures & Human-in-the-Loop workflows
│   ├── create_agent.py       # High-level agent with middleware (wrap_tool_call) & HITL approval
│   └── manual_agents.py      # Low-level manual agent loop, tool dispatch & state management
│
├── rag/                      # Retrieval-Augmented Generation module
│   ├── main.py               # Interactive CLI RAG chatbot grounded on vector documents
│   ├── create_data.py        # PDF document loader, chunker, embeddings & ChromaDB storage
│   ├── test.py               # Benchmark & compare Similarity Search vs. MMR retrieval
│   ├── requirements.txt      # RAG module dependencies
│   ├── README.md             # Dedicated RAG documentation
│   └── documents-loaders/    # Sample data and document loading experiments
│
├── runnables/                # LangChain Expression Language (LCEL) pipelines
│   ├── sequencerunnable.py   # Basic sequential chain (prompt | model | parser)
│   ├── parallelrunnable.py   # Multi-branch concurrent execution (RunnableParallel & RunnableLambda)
│   ├── runnablepassthrough.py# Advanced data routing (RunnablePassthrough & RunnableParallel)
│   └── requirements.txt      # Runnables dependencies
│
├── tools/                    # Tool definitions, binding & search integrations
│   ├── toolcalling.py        # Custom @tool creation, LLM tool binding & invocation
│   ├── newssummarizer.py     # Live search integration (Tavily) chained with summarizer LLM
│   └── requirements.txt      # Tools dependencies
│
├── requirements.txt          # Root project dependencies
├── .env                      # API keys & environment configuration (git-ignored)
└── README.md                 # Workspace documentation
```

---

## 🚀 Core Architecture & Modules

### 1. 🤖 [AI Agents & Human-in-the-Loop (`agents/`)](agents/)
Covers autonomous AI agent implementations, function calling, tool orchestration, and interactive human verification.

- **Key Implementations**:
  - **High-Level Agent (`create_agent.py`)**: Built with LangChain's `create_agent` and modern middleware (`@wrap_tool_call`) for real-time Human-in-the-Loop (HITL) tool execution approval.
  - **Manual Agent Loop (`manual_agents.py`)**: A transparent, low-level implementation of the ReAct/agent loop that handles message state (`HumanMessage`, `AIMessage`, `ToolMessage`, `SystemMessage`), tool dispatch, denial handling, and multi-turn conversation.
  - **Integrated Tools**:
    - `get_weather`: Live city weather via **OpenWeatherMap API**.
    - `get_news`: Live city news via **Tavily Search API**.
- **Quick Run**:
  ```bash
  # Run modern agent with middleware & HITL
  python agents/create_agent.py

  # Run low-level manual agent loop
  python agents/manual_agents.py
  ```

---

### 2. 🔍 [Retrieval-Augmented Generation (`rag/`)](rag/README.md)
Covers document ingestion, vector stores, semantic search, and context-grounded conversational chatbots.

- **Capabilities**: PDF parsing (`PyPDFLoader`), recursive text splitting (`RecursiveCharacterTextSplitter`), dense vector embeddings with **Mistral AI Embeddings**, persistent storage with **ChromaDB**, and comparative retrieval (**Similarity Search** vs. **Maximal Marginal Relevance / MMR**).
- **Quick Run**:
  ```bash
  # 1. Ingest and embed PDF documents into ChromaDB
  python rag/create_data.py

  # 2. Test and compare retrieval algorithms
  python rag/test.py

  # 3. Launch interactive CLI RAG assistant
  python rag/main.py
  ```

---

### 3. ⚡ [LangChain Expression Language & Runnables (`runnables/`)](runnables/)
Covers LCEL composability, sequential pipelines, parallel branches, and data passthrough techniques.

- **Key Implementations**:
  - **Sequential Chain (`sequencerunnable.py`)**: Direct pipeline composition using the pipe operator (`prompt | model | parser`).
  - **Parallel Execution (`parallelrunnable.py`)**: Concurrent multi-prompt branches with `RunnableParallel` and input mapping via `RunnableLambda`.
  - **Passthrough & Chaining (`runnablepassthrough.py`)**: Generating intermediate artifacts (e.g., code) and passing both raw outputs and downstream explanations using `RunnablePassthrough`.
- **Quick Run**:
  ```bash
  python runnables/sequencerunnable.py
  python runnables/parallelrunnable.py
  python runnables/runnablepassthrough.py
  ```

---

### 4. 🛠️ [Tools & Tool Calling (`tools/`)](tools/)
Covers tool definitions, schema binding, and external API tool chains.

- **Key Implementations**:
  - **Tool Calling Basics (`toolcalling.py`)**: Defining typed tools with `@tool`, binding tools to chat models (`llm.bind_tools`), inspecting tool calls, and passing execution results back to the LLM.
  - **Live Web Search & News Summarizer (`newssummarizer.py`)**: Integrating `TavilySearchResults` with an LCEL summarization chain to fetch and summarize real-time web news into bullet points.
- **Quick Run**:
  ```bash
  python tools/toolcalling.py
  python tools/newssummarizer.py
  ```

---

## 🛠️ Quick Start & Setup

### 1. Set Up Virtual Environment

Using `uv` (recommended):
```bash
uv venv
source .venv/bin/activate
```

Using standard `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

Install the comprehensive root requirements:
```bash
pip install -r requirements.txt
pip install tavily-python rich
```

### 3. Environment Configuration

Create a `.env` file at the project root with your API credentials:

```env
# Mistral AI (Chat & Embeddings)
MISTRAL_API_KEY=your_mistral_api_key_here

# OpenWeather (Weather Tool)
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Tavily (Live Web Search & News Tool)
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## 📌 Architecture & Engineering Roadmap

- [x] **Retrieval-Augmented Generation (RAG)**: Ingestion, ChromaDB vector store, MMR search, grounded chatbot.
- [x] **LangChain Expression Language (LCEL)**: Sequential chains, `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`.
- [x] **Tools & External APIs**: Custom tool definition, model tool-binding, Tavily Search & OpenWeather integrations.
- [x] **AI Agents & Human-in-the-Loop**: Tool calling agents, middleware interception, manual vs. high-level agent loops.
- [ ] **LangGraph Workflows**: Stateful multi-agent graphs, cycles, and persistent checkpointing.
- [ ] **Fine-Tuning & Quantization**: Parameter-efficient fine-tuning (PEFT/LoRA) workflows.
- [ ] **Multimodal AI**: Vision & audio pipelines.
