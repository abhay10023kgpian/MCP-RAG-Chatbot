# 🏗️ ARCHITECTURE — Multi-Server MCP RAG Chatbot

> Complete system design, request flows, component documentation, and tech stack reference.
> Use this document to understand **what each piece does**, **how data flows**, and **how the system is wired together**.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Tech Stack Deep-Dive](#2-tech-stack-deep-dive)
3. [Project File Tree](#3-project-file-tree)
4. [Architecture Diagram](#4-architecture-diagram)
5. [Request Flow — Step by Step](#5-request-flow--step-by-step)
6. [Component Documentation (File-by-File)](#6-component-documentation-file-by-file)
7. [Function Call Graph](#7-function-call-graph)
8. [RAG Pipeline — Embedding to Answer](#8-rag-pipeline--embedding-to-answer)
9. [LangGraph — Chatbot State Machine](#9-langgraph--chatbot-state-machine)
10. [Streaming (SSE) Flow](#10-streaming-sse-flow)
11. [Legacy MCP Servers (server/)](#11-legacy-mcp-servers-server)
12. [Environment Variables](#12-environment-variables)
13. [Deployment on Render](#13-deployment-on-render)
14. [How to Run — Commands Reference](#14-how-to-run--commands-reference)
15. [Design Decisions & Trade-offs](#15-design-decisions--trade-offs)

---

## 1. System Overview

This project is a **multi-tool chatbot** that:

1. **Receives user questions** via a Streamlit chat UI (or REST API)
2. **Routes to an LLM** (Groq — Llama 3.3 70B) which decides if a tool is needed
3. **Calls direct LangChain tools** if needed (RAG knowledge retrieval or calculator)
4. **Retrieves relevant documents** from a ChromaDB vector store (Google Gemini embeddings)
5. **Synthesizes an answer** using the retrieved context
6. **Returns "no data available"** if the query topic isn't in the knowledge base

### What makes this different from a basic chatbot?

| Feature | Basic Chatbot | This Project |
|---------|--------------|-------------|
| Knowledge | General LLM knowledge only | Grounded in YOUR documents |
| Tools | None | Direct LangChain tools (extensible) |
| Memory | Stateless | Thread-based conversation memory |
| Architecture | Monolithic | Modular (client/api/frontend) |
| Tool Calling | Manual if/else | LLM decides automatically via LangGraph |
| Streaming | None | SSE token-by-token streaming |

### Current Tool Architecture

> **Important**: The production API uses **direct `@tool` functions** (`client/rag_tool.py`, `client/math_tools.py`) — these run in-process, with no MCP subprocess overhead. The `server/` directory contains legacy MCP server implementations that are still functional for standalone/CLI testing via `client/mcp_client.py`.

---

## 2. Tech Stack Deep-Dive

### Provider Layer

| Technology | Role | Why This Choice |
|-----------|------|----------------|
| **Groq** | LLM inference | Free tier, ultra-fast (LPU hardware), Llama 3.3 70B |
| **Google Gemini** (`gemini-embedding-2`) | Embedding model | Free tier (1500 req/day), 768-dim vectors, lightweight (no PyTorch!) |

### Framework Layer

| Technology | Role | Why This Choice |
|-----------|------|----------------|
| **LangChain** | LLM abstraction | Unified interface for Groq/embeddings/vector stores |
| **LangGraph** | Chatbot orchestration | Stateful graph with conditional tool routing |
| **FastMCP** | MCP tool server framework | Decorator-based tool definition (used in `server/` legacy path) |
| **langchain-mcp-adapters** | MCP client bridge | Connects LangChain tools to MCP servers (used in `client/mcp_client.py` legacy path) |

### Data Layer

| Technology | Role | Why This Choice |
|-----------|------|----------------|
| **ChromaDB** | Vector database | File-based (no server needed), good for local dev |
| **RecursiveCharacterTextSplitter** | Text chunking | Smart splitting on natural boundaries |

### Application Layer

| Technology | Role | Why This Choice |
|-----------|------|----------------|
| **FastAPI** | REST API backend | Async-native, auto-docs, Pydantic validation |
| **Streamlit** | Chat UI frontend | Rapid prototyping, built-in chat components |
| **Uvicorn** | ASGI server | Production-grade async server for FastAPI |

### Utility Layer

| Technology | Role | Why This Choice |
|-----------|------|----------------|
| **python-dotenv** | Config management | Standard .env loading |
| **Pydantic** | Data validation | Request/response schemas |

---

## 3. Project File Tree

```
mcp_rag_chatbot/
├── api/
│   ├── __init__.py              # API package marker
│   └── server.py                # FastAPI REST backend (endpoints, lifespan, SSE streaming)
│
├── client/
│   ├── __init__.py              # Client package marker
│   ├── chatbot.py               # LangGraph stateful chatbot (agent_node, tool routing)
│   ├── config.py                # Centralized configuration loader
│   ├── math_tools.py            # Direct @tool: calculator_add, calculator_multiply
│   ├── mcp_client.py            # Legacy MCP client (stdio subprocess connector)
│   └── rag_tool.py              # Direct @tool: retrieve_from_knowledge_base (ChromaDB)
│
├── server/                      # ⚠️ Legacy MCP servers (not used by production API)
│   ├── __init__.py              # Server package marker
│   ├── math_server.py           # FastMCP math tool server
│   └── rag_server.py            # FastMCP RAG tool server
│
├── frontend/
│   └── app.py                   # Streamlit chat UI (SSE streaming client)
│
├── storage/
│   ├── notes.txt                # Knowledge base source document
│   ├── physics_notes.pdf        # PDF source (not auto-embedded)
│   └── theory_motion_bodies_rag_testing.txt  # Large test document
│
├── chroma_db/                   # ChromaDB persistent storage (generated by embed_documents.py)
│
├── embed_documents.py           # One-time document embedding pipeline
├── start.py                     # Production startup script (Render deployment)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker container definition
├── docker-compose.yml           # Docker Compose orchestration
├── render.yaml                  # Render deployment config
├── .env                         # Environment variables (gitignored)
└── .env.example                 # Env template
```

---

## 4. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                              │
│  ┌──────────────────┐     ┌──────────────────────────────────────┐  │
│  │  Streamlit UI     │     │  Any HTTP Client (curl, Postman)    │  │
│  │  frontend/app.py  │     │  Direct API access                  │  │
│  └────────┬─────────┘     └─────────────────┬────────────────────┘  │
│           │ POST /chat/stream (SSE)          │ POST /chat (JSON)    │
└───────────┼──────────────────────────────────┼──────────────────────┘
            │                                  │
            ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API LAYER (FastAPI)                           │
│  api/server.py                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Endpoints:                                                     │  │
│  │   GET  /health        → Server status + connected tools       │  │
│  │   GET  /tools         → List available tool names             │  │
│  │   POST /chat          → Invoke chatbot, return full response  │  │
│  │   POST /chat/stream   → SSE token-by-token streaming         │  │
│  │                                                                │  │
│  │ Lifespan:                                                      │  │
│  │   startup  → initialize_tools() → create_chatbot()            │  │
│  │   shutdown → cleanup                                          │  │
│  │                                                                │  │
│  │ app_state:                                                     │  │
│  │   chatbot, make_config, tools, named_tools, is_ready          │  │
│  └─────────────────────────────┬──────────────────────────────────┘  │
└────────────────────────────────┼────────────────────────────────────┘
                                 │ calls create_chatbot(tools)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER (LangGraph)                   │
│  client/chatbot.py                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │   START → [agent_node] → {has tool calls?}                    │  │
│  │                             │            │                     │  │
│  │                            YES          NO                     │  │
│  │                             │            │                     │  │
│  │                             ▼            ▼                     │  │
│  │                       [ToolNode]       [END]                   │  │
│  │                             │                                  │  │
│  │                             └──→ [agent_node] → [END]         │  │
│  │                              (synthesize with tool results)    │  │
│  │                                                                │  │
│  │   MemorySaver: Thread-based conversation persistence          │  │
│  └─────────────────────────────┬──────────────────────────────────┘  │
└────────────────────────────────┼────────────────────────────────────┘
                                 │ ToolNode invokes tools directly
                                 │ (in-process, no subprocess)
                                 ▼
┌──────────────────────────┐   ┌──────────────────────────────────────┐
│  DIRECT TOOL 1           │   │  DIRECT TOOL 2                       │
│  client/math_tools.py    │   │  client/rag_tool.py                  │
│                          │   │                                      │
│  @tool functions:        │   │  @tool function:                     │
│  • calculator_add(a, b)  │   │  • retrieve_from_knowledge_base(q)  │
│  • calculator_multiply   │   │                                      │
│    (a, b)                │   │  Calls _get_vector_store()           │
│                          │   │       │ (lazy singleton)             │
│  Pure functions,         │   │       ▼                              │
│  no external deps        │   │  ┌────────────────────────────┐     │
│                          │   │  │  Google Gemini Embeddings  │     │
│                          │   │  │  Embed query → vector      │     │
│                          │   │  └─────────────┬──────────────┘     │
│                          │   │                │                     │
│                          │   │                ▼                     │
│                          │   │  ┌────────────────────────────┐     │
│                          │   │  │  ChromaDB Vector Store     │     │
│                          │   │  │  Similarity search (k=3)   │     │
│                          │   │  │  Score threshold ≥ 0.3     │     │
│                          │   │  └────────────────────────────┘     │
└──────────────────────────┘   └──────────────────────────────────────┘
```

---

## 5. Request Flow — Step by Step

### Scenario: User asks *"What is Newton's second law?"*

```
Step 1  │ USER types message in Streamlit UI
        │ frontend/app.py sends POST /chat/stream to FastAPI (SSE)
        │
Step 2  │ FastAPI receives ChatRequest{query, thread_id}
        │ api/server.py → chat_stream() starts event_generator()
        │
Step 3  │ LangGraph: agent_node runs
        │ Groq LLM sees system prompt + user message + tool schemas
        │ LLM decides: "I should search the knowledge base"
        │ Returns AIMessage with tool_calls:
        │   [{name: "retrieve_from_knowledge_base", args: {query: "Newton's second law"}}]
        │
Step 4  │ LangGraph: should_continue() → "tools"
        │ ToolNode executes the direct @tool function (in-process)
        │
Step 5  │ client/rag_tool.py → retrieve_from_knowledge_base():
        │   a) Calls _get_vector_store() (lazy singleton — initialized once)
        │   b) Embeds query with Google Gemini API
        │   c) Searches ChromaDB for similar vectors (k=3)
        │   d) Filters by relevance score (≥ 0.3)
        │   e) Returns JSON: {found: true, documents: ["...", "..."], scores: [0.87, 0.72]}
        │
Step 6  │ LangGraph: ToolNode → agent_node (loop back)
        │ ToolMessage with retrieval results added to conversation
        │
Step 7  │ LangGraph: agent_node runs AGAIN
        │ Groq LLM now sees: system prompt + user Q + tool results
        │ System prompt: "base your answer PRIMARILY on retrieved documents"
        │ LLM synthesizes answer from the retrieved chunks
        │ Returns AIMessage with content (final answer)
        │ SSE streams each token as {type: "content", text: "..."} to frontend
        │
Step 8  │ LangGraph: should_continue() → END (no more tool calls)
        │
Step 9  │ SSE sends {type: "done"} to close the stream
        │
Step 10 │ Streamlit displays:
        │   - AI response text (streamed token-by-token)
        │   - 🔧 Tool badge: "retrieve_from_knowledge_base"
        │   - 📄 Expandable source documents
```

### Scenario: User asks *"What is quantum entanglement?"* (NOT in knowledge base)

```
Steps 1-4 │ Same as above...
           │
Step 5     │ client/rag_tool.py:
           │   Searches ChromaDB → all scores < 0.3 threshold
           │   Returns: {found: false, documents: [], message: "No relevant documents..."}
           │
Steps 6-7  │ agent_node sees {found: false}
           │ System prompt says: "If found is false → say 'I don't have specific
           │   information on that in my knowledge base, but here's what I can share...'"
           │ LLM responds with brief general knowledge answer, clearly distinguished
           │
Steps 8-10 │ Response delivered to user via SSE
```

### Scenario: User asks *"What is 25 + 37?"*

```
Step 3  │ Groq LLM sees the math question
        │ Decides to use calculator_add tool
        │ Returns: tool_calls: [{name: "calculator_add", args: {a: 25, b: 37}}]
        │
Step 4  │ ToolNode calls client/math_tools.py → calculator_add(25, 37)
        │ Returns: 62  (pure function, no external calls)
        │
Step 7  │ LLM synthesizes: "25 + 37 = 62"
```

### Scenario: User says *"Hello, how are you?"*

```
Step 3  │ Groq LLM sees a greeting
        │ No tool call needed — responds directly
        │ Returns: AIMessage with content "Hello! I'm doing great..."
        │
Step 4  │ should_continue() → END (no tool calls)
        │ Skips ToolNode entirely
```

---

## 6. Component Documentation (File-by-File)

---

### 📄 `api/server.py` — FastAPI REST Backend

**Purpose**: HTTP API wrapping the chatbot for the frontend. Manages tool initialization and chatbot lifecycle.

#### Global State

```python
app_state = {
    "chatbot": None,       # Compiled LangGraph graph
    "make_config": None,   # Config factory function
    "tools": None,         # List of LangChain Tool objects
    "named_tools": None,   # Dict: tool_name → Tool object
    "is_ready": False,     # True after initialization completes
}
```

#### Functions

| Function | Signature | What It Does | Called By |
|----------|-----------|-------------|----------|
| `initialize_tools()` | `async def initialize_tools()` | Loads direct tools (`rag_tool`, `math_tools`), calls `create_chatbot()`, populates `app_state` | `lifespan()` via `asyncio.create_task` |
| `lifespan(app)` | `async def lifespan(app: FastAPI)` | FastAPI lifespan handler — spawns `initialize_tools()` as background task on startup | FastAPI framework |

#### Endpoints

| Endpoint | Method | Handler | Request | Response |
|----------|--------|---------|---------|----------|
| `/health` | GET | `health_check()` | — | `{status, tools_connected, tool_names}` |
| `/tools` | GET | `list_tools()` | — | `{tools: [{name, description}]}` |
| `/chat` | POST | `chat(request)` | `ChatRequest{query, thread_id}` | `ChatResponse{response, tools_used, source_docs, thread_id}` |
| `/chat/stream` | POST | `chat_stream(request)` | `ChatRequest{query, thread_id}` | SSE stream of `{type, ...}` events |

#### Request/Response Schemas

```python
class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    tools_used: list[str] = []
    source_docs: list[str] = []
    thread_id: str = ""
```

#### SSE Event Types (from `/chat/stream`)

| Event Type | Payload | When |
|-----------|---------|------|
| `tool_start` | `{type: "tool_start", tool: "<name>"}` | When a tool begins execution |
| `source_docs` | `{type: "source_docs", docs: [...]}` | When RAG tool returns with `found: true` |
| `content` | `{type: "content", text: "<token>"}` | Each LLM token (from nodes tagged `stream_response`) |
| `done` | `{type: "done"}` | Stream complete |
| `error` | `{type: "error", message: "<msg>"}` | Error during streaming |

#### Call Chain

```
server.py startup:
  lifespan() → asyncio.create_task(initialize_tools())
    initialize_tools()
      → imports: rag_tool.retrieve_from_knowledge_base
      → imports: math_tools.calculator_add, calculator_multiply
      → calls: chatbot.create_chatbot(tools)
      → populates: app_state

POST /chat:
  chat(request)
    → app_state["chatbot"].ainvoke({messages: [HumanMessage]}, config)
    → extracts: final response, tools_used, source_docs from messages
    → returns: ChatResponse

POST /chat/stream:
  chat_stream(request)
    → event_generator() (async generator)
      → app_state["chatbot"].astream_events({messages: [HumanMessage]}, config, version="v2")
      → yields SSE events for: tool_start, tool_end, chat_model_stream, chain_end
```

#### Imports

| From | What | Purpose |
|------|------|---------|
| `client.rag_tool` | `retrieve_from_knowledge_base` | RAG retrieval tool |
| `client.math_tools` | `calculator_add`, `calculator_multiply` | Math tools |
| `client.chatbot` | `create_chatbot` | LangGraph chatbot factory |

---

### 📄 `client/chatbot.py` — LangGraph Chatbot Engine

**Purpose**: The brain — builds a stateful LangGraph chatbot that decides when to use tools.

#### State Schema

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Auto-appending message list
    category: str                                          # Message category (reserved)
```

#### Functions

| Function | Signature | What It Does | Returns |
|----------|-----------|-------------|---------|
| `create_chatbot(tools)` | `async def create_chatbot(tools: list = None)` | Builds LangGraph: initializes Groq LLM, binds tools, creates `agent_node`, `ToolNode`, conditional routing, compiles with `MemorySaver` | `(compiled_graph, make_config)` |
| `run_interactive()` | `async def run_interactive()` | CLI interactive mode for testing. Connects via MCP (`get_mcp_tools()`), creates chatbot, runs input loop | None |

#### Internal Nodes (created inside `create_chatbot`)

| Node | Name in Graph | Role | Input | Output |
|------|--------------|------|-------|--------|
| `agent_node` | `"agent"` | Invokes Groq LLM with system prompt + tools bound | `ChatState` (messages) | `{messages: [AIMessage]}` (may contain `tool_calls`) |
| `ToolNode(tools)` | `"tools"` | Executes tool calls from AIMessage | `AIMessage.tool_calls` | `{messages: [ToolMessage]}` (tool results) |

#### Routing Logic

```python
def should_continue(state: ChatState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END
```

#### Graph Edges

```
START ──→ "agent"
"agent" ──(conditional)──→ "tools"  (if tool_calls present)
"agent" ──(conditional)──→ END      (if no tool_calls)
"tools" ──→ "agent"                 (loop back for synthesis)
```

#### System Prompt (inside `create_chatbot`)

```
You are a knowledgeable assistant grounded in a specific knowledge base.

TOOL USAGE RULES:
1. For ANY factual question, MUST call retrieve_from_knowledge_base FIRST.
2. For math calculations, use the calculator tools.
3. For greetings and small talk, respond naturally without tools.

ANSWERING RULES:
- If tool returns relevant documents → base answer PRIMARILY on those documents.
- If tool returns found: false → say "I don't have specific information..."
- NEVER present general knowledge as if it came from the knowledge base.
```

#### LLM Configuration

```python
llm = ChatGroq(
    api_key=GROQ_API_KEY,     # from .env
    model=GROQ_MODEL,         # default: "llama-3.3-70b-versatile"
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools).with_config({"tags": ["stream_response"]})
```

> The `"stream_response"` tag is used by `api/server.py`'s streaming endpoint to identify which LLM tokens to forward via SSE.

#### Call Chain

```
create_chatbot(tools):
  → ChatGroq(api_key, model, temperature=0)
  → llm.bind_tools(tools).with_config(tags=["stream_response"])
  → defines agent_node(state) → llm_with_tools.ainvoke([system_prompt] + messages)
  → defines should_continue(state) → "tools" | END
  → StateGraph(ChatState)
      .add_node("agent", agent_node)
      .add_node("tools", ToolNode(tools))
      .add_edge(START, "agent")
      .add_conditional_edges("agent", should_continue)
      .add_edge("tools", "agent")
  → graph.compile(checkpointer=MemorySaver())
  → returns (compiled, make_config)

run_interactive():
  → client.mcp_client.get_mcp_tools()  # Legacy MCP path
  → create_chatbot(tools)
  → loop: input → chatbot.ainvoke() → print response
```

---

### 📄 `client/rag_tool.py` — Direct RAG Retrieval Tool

**Purpose**: In-process `@tool` that searches ChromaDB for documents matching a query. Replaces the MCP-based `server/rag_server.py` subprocess.

#### Functions

| Function | Signature | What It Does | Returns |
|----------|-----------|-------------|---------|
| `_get_vector_store()` | `def _get_vector_store()` | Lazy singleton — initializes ChromaDB + Google Gemini embeddings on first call, caches globally | `Chroma` instance |
| `retrieve_from_knowledge_base(query)` | `@tool def retrieve_from_knowledge_base(query: str) -> str` | Searches ChromaDB (k=3), filters by relevance ≥ 0.3, returns JSON | JSON string |

#### Configuration (from .env)

| Constant | Default |
|----------|---------|
| `CHROMA_DIR` | `<project_root>/chroma_db` |
| `COLLECTION_NAME` | `rag_knowledge_base` |
| `EMBEDDING_MODEL` | `models/gemini-embedding-2` |
| `GOOGLE_API_KEY` | (required) |

#### Return Schema

```json
// When documents are found:
{
  "found": true,
  "query": "Newton's second law",
  "documents": ["Force equals mass times...", "F=ma is..."],
  "scores": [0.87, 0.72],
  "num_results": 2
}

// When no documents match:
{
  "found": false,
  "query": "quantum entanglement",
  "documents": [],
  "num_results": 0,
  "message": "No relevant documents found in the knowledge base for this query."
}

// On error:
{
  "found": false,
  "query": "...",
  "documents": [],
  "num_results": 0,
  "error": "Retrieval error: ..."
}
```

#### Call Chain

```
retrieve_from_knowledge_base(query):
  → _get_vector_store()
      → GoogleGenerativeAIEmbeddings(model, api_key)
      → Chroma(collection_name, embedding_function, persist_directory)
      → sanity check: collection.count()
      → caches in global _vector_store
  → vector_store.similarity_search_with_relevance_scores(query, k=3)
  → filters results where score >= 0.3
  → returns json.dumps({found, query, documents, scores, num_results})
```

---

### 📄 `client/math_tools.py` — Direct Calculator Tools

**Purpose**: Pure-function `@tool` wrappers for basic arithmetic. Replaces `server/math_server.py`.

#### Functions

| Function | Signature | What It Does | Returns |
|----------|-----------|-------------|---------|
| `calculator_add(a, b)` | `@tool def calculator_add(a: int, b: int) -> int` | Returns `a + b` | `int` |
| `calculator_multiply(a, b)` | `@tool def calculator_multiply(a: int, b: int) -> int` | Returns `a * b` | `int` |

No external dependencies. No state. Pure functions.

---

### 📄 `client/config.py` — Configuration Hub

**Purpose**: Single source of truth for all environment variables. Loads from `.env`.

#### Constants

| Constant | Source | Default |
|----------|--------|---------|
| `GROQ_API_KEY` | `.env` | (required) |
| `GROQ_MODEL` | `.env` | `llama-3.3-70b-versatile` |
| `GOOGLE_API_KEY` | `.env` | (required) |
| `EMBEDDING_MODEL` | `.env` | `models/gemini-embedding-2` |
| `CHROMA_DIR` | `.env` | `<project_root>/chroma_db` |
| `COLLECTION_NAME` | `.env` | `rag_knowledge_base` |
| `STORAGE_DIR` | computed | `<project_root>/storage` |
| `API_HOST` | `.env` | `0.0.0.0` |
| `API_PORT` | `.env` | `8000` |
| `PYTHON_EXECUTABLE` | `sys.executable` | current interpreter |
| `RAG_SERVER_PATH` | computed | `<project_root>/server/rag_server.py` |
| `MATH_SERVER_PATH` | computed | `<project_root>/server/math_server.py` |

#### Functions

| Function | Signature | What It Does | Returns |
|----------|-----------|-------------|---------|
| `validate_config()` | `def validate_config()` | Checks `GROQ_API_KEY` and `GOOGLE_API_KEY` are set. Calls `sys.exit(1)` if missing. | `True` |
| `print_config()` | `def print_config()` | Prints config table with masked API keys for debugging. | None |

---

### 📄 `client/mcp_client.py` — Legacy Multi-Server MCP Client

**Purpose**: Connects to MCP tool servers via stdio subprocesses. **Used only by `run_interactive()` in CLI mode** — the production API uses direct tools instead.

#### Functions

| Function | Signature | What It Does | Returns |
|----------|-----------|-------------|---------|
| `get_mcp_server_config()` | `def get_mcp_server_config() -> dict` | Returns config dict for MCP servers with paths and env | `dict` |
| `get_mcp_tools()` | `async def get_mcp_tools()` | Connects to all MCP servers, discovers tools | `(client, tools, named_tools)` |

#### Server Config Structure

```python
{
    "rag_knowledge_server": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["<project_root>/server/rag_server.py"],
        "env": {**os.environ, "PYTHONIOENCODING": "utf-8"}
    },
    "math_tools_server": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["<project_root>/server/math_server.py"],
        "env": {**os.environ, "PYTHONIOENCODING": "utf-8"}
    }
}
```

#### Call Chain

```
get_mcp_tools():
  → get_mcp_server_config()
  → MultiServerMCPClient(servers)
  → client.get_tools()   # spawns subprocesses, discovers tools
  → returns (client, tools, named_tools)
```

> **Important**: The `client` object must stay alive! If garbage collected, subprocesses die.

---

### 📄 `frontend/app.py` — Streamlit Chat UI

**Purpose**: User-facing chat interface. Communicates with the API via SSE streaming.

#### Configuration

| Constant | Default | Source |
|----------|---------|--------|
| `API_URL` | `http://127.0.0.1:8000` | `API_URL` env var |
| `THREAD_ID` | `streamlit-session` | hardcoded |

#### Features

| Feature | Implementation |
|---------|---------------|
| Chat input | `st.chat_input()` |
| Message history | `st.session_state["message_history"]` (list of dicts) |
| Streaming | Reads SSE from `POST /chat/stream`, renders tokens incrementally |
| Tool badges | Shown below AI responses when tools are used |
| Source docs | Expandable `st.expander` with retrieved chunks (truncated at 500 chars) |
| API health | Sidebar — `GET /health`, shows connected tool count/names |
| Suggested queries | Sidebar with example questions |

#### Message History Entry Schema

```python
{
    "role": "user" | "assistant",
    "content": str,
    "tools_used": list[str],    # assistant only
    "source_docs": list[str],   # assistant only
}
```

#### Call Chain

```
Page load:
  → requests.get(API_URL + "/health")
  → displays sidebar status

User sends message:
  → appends to st.session_state["message_history"]
  → requests.post(API_URL + "/chat/stream", json={query, thread_id}, stream=True)
  → iterates SSE lines:
      "tool_start"  → updates status_placeholder
      "source_docs" → collects docs
      "content"     → appends to full_response, updates message_placeholder
      "done"        → breaks
      "error"       → displays error
  → appends assistant message to history
  → renders tool badges + source docs expander
```

---

### 📄 `embed_documents.py` — Document Embedding Pipeline

**Purpose**: One-time script to process raw `.txt` files from `storage/` into ChromaDB vectors.

#### Functions

| Function | Signature | What It Does | Returns |
|----------|-----------|-------------|---------|
| `load_text_file(filepath)` | `def load_text_file(filepath: str) -> str` | Reads a `.txt` file with UTF-8 encoding | `str` |
| `split_text(text)` | `def split_text(text: str) -> list[str]` | Splits text into chunks (1000 chars, 50 overlap) using `RecursiveCharacterTextSplitter` | `list[str]` |
| `main()` | `def main()` | CLI entry — parses args, loads files, chunks, embeds in batches, stores in ChromaDB | None |

#### Chunking Strategy

```
RecursiveCharacterTextSplitter:
  chunk_size = 1000 chars
  chunk_overlap = 50 chars
  separators = ["\n\n", "\n", ". ", ", ", " "]
  
  Tries paragraph splits first, then sentences, then words.
  Overlap ensures context isn't lost at chunk boundaries.
```

#### Batch Processing (Rate Limit Protection)

```
batch_size = 80 chunks per batch
sleep = 65 seconds between batches
reason: Google API free tier = 100 requests/minute
```

#### Call Chain

```
main():
  → argparse: --file (optional)
  → validates GOOGLE_API_KEY
  → glob storage/*.txt (or specific file)
  → for each file:
      → load_text_file(filepath)
      → split_text(text)
      → collect chunks + metadata (source, chunk_index)
  → GoogleGenerativeAIEmbeddings(model, api_key)
  → batch loop (80 chunks per batch):
      → Chroma.from_texts() (first batch) or vector_store.add_texts() (subsequent)
      → time.sleep(65) between batches
  → verify: collection.get() → prints vector count + dimensions
```

#### CLI Usage

```bash
python embed_documents.py                    # Embed all .txt files in storage/
python embed_documents.py --file myfile.txt  # Embed a specific file
```

---

### 📄 `start.py` — Production Startup Script

**Purpose**: Entry point for Render/cloud deployment. Ensures ChromaDB exists, then starts uvicorn.

#### Functions

| Function | Signature | What It Does |
|----------|-----------|-------------|
| `run_embedding()` | `def run_embedding()` | Checks if `chroma_db/chroma.sqlite3` exists. If missing, runs `embed_documents.py` as subprocess. |
| `start_server()` | `def start_server()` | Starts uvicorn via `os.execvp` (replaces current process). Uses `PORT` env (Render) or `API_PORT`. |

#### Call Chain

```
main:
  → run_embedding()
      → checks chroma_db/chroma.sqlite3
      → if missing: subprocess.run([python, embed_documents.py])
  → start_server()
      → os.execvp(python, ["-m", "uvicorn", "api.server:app", "--host", host, "--port", port])
```

---

## 7. Function Call Graph

Complete call chain from user input to response:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STARTUP CALL CHAIN                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  start.py:main()                                                        │
│    ├── run_embedding()                                                  │
│    │     └── subprocess: embed_documents.py:main()                     │
│    │           ├── load_text_file(path) × N files                      │
│    │           ├── split_text(text) × N files                          │
│    │           ├── GoogleGenerativeAIEmbeddings()                       │
│    │           └── Chroma.from_texts() / add_texts() in batches        │
│    │                                                                    │
│    └── start_server()                                                   │
│          └── os.execvp → uvicorn api.server:app                        │
│                └── api/server.py:lifespan(app)                         │
│                      └── asyncio.create_task(initialize_tools())       │
│                            ├── imports: rag_tool, math_tools           │
│                            ├── chatbot.create_chatbot(tools)           │
│                            │     ├── ChatGroq()                        │
│                            │     ├── llm.bind_tools(tools)             │
│                            │     ├── StateGraph(ChatState)             │
│                            │     │     ├── add_node("agent")           │
│                            │     │     ├── add_node("tools", ToolNode) │
│                            │     │     └── add_conditional_edges()     │
│                            │     ├── graph.compile(MemorySaver())      │
│                            │     └── returns (compiled, make_config)   │
│                            └── populates app_state                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    REQUEST CALL CHAIN (POST /chat)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  frontend/app.py                                                        │
│    └── requests.post("/chat/stream", json={query, thread_id})          │
│                                                                         │
│  api/server.py:chat_stream(request)                                    │
│    └── event_generator()                                                │
│          └── chatbot.astream_events({messages: [HumanMessage]}, config) │
│                │                                                        │
│                ├── agent_node(state)                                    │
│                │     └── llm_with_tools.ainvoke([system_prompt]+msgs)   │
│                │           └── Groq API call → AIMessage               │
│                │                                                        │
│                ├── should_continue(state)                               │
│                │     └── checks last_message.tool_calls                 │
│                │           ├── has calls → "tools"                      │
│                │           └── no calls  → END                          │
│                │                                                        │
│                └── ToolNode(tools).invoke(state)                        │
│                      ├── retrieve_from_knowledge_base(query)           │
│                      │     ├── _get_vector_store()                     │
│                      │     │     ├── GoogleGenerativeAIEmbeddings()    │
│                      │     │     └── Chroma(collection, embedding)     │
│                      │     └── similarity_search_with_relevance_scores │
│                      │                                                  │
│                      ├── calculator_add(a, b)     → a + b              │
│                      └── calculator_multiply(a, b) → a * b             │
│                                                                         │
│  → SSE events streamed back to frontend                                │
│  → frontend renders tokens incrementally                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│               REQUEST CALL CHAIN (POST /chat — non-streaming)           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  api/server.py:chat(request)                                           │
│    └── chatbot.ainvoke({messages: [HumanMessage]}, config)             │
│          └── (same agent_node → tools → agent_node flow as above)      │
│    └── extracts from result["messages"]:                               │
│          ├── final_response  = messages[-1].content                    │
│          ├── tools_used      = [tc["name"] for AIMessage.tool_calls]   │
│          └── source_docs     = json.loads(ToolMessage.content)         │
│                                  → filters {found: true}.documents     │
│    └── returns ChatResponse{response, tools_used, source_docs}         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. RAG Pipeline — Embedding to Answer

### Phase 1: Document Embedding (One-Time via `embed_documents.py`)

```
Raw Text Files                    Vector Store
storage/*.txt                     chroma_db/
     │                                 ▲
     │ load_text_file()                │
     ▼                                 │
 Full Text                             │
     │                                 │
     │ RecursiveCharacterTextSplitter   │
     │ (1000 chars, 50 overlap)         │
     ▼                                 │
 Text Chunks                           │
 ["chunk1", "chunk2", ...]             │
     │                                 │
     │ Google Gemini Embeddings        │
     │ gemini-embedding-2              │
     │ (768 dimensions)               │
     ▼                                 │
 Vectors                               │
 [[0.12, -0.34, ...], ...]            │
     │                                 │
     │ Chroma.from_texts() / add_texts │
     │ (80 chunks/batch, 65s sleep)    │
     └─────────────────────────────────┘
```

### Phase 2: Query & Retrieval (Per User Query via `client/rag_tool.py`)

```
User Query: "What is acceleration?"
     │
     │ Google Gemini Embeddings
     │ (same model as embedding phase)
     ▼
 Query Vector [0.45, -0.12, ...]  (768 dims)
     │
     │ ChromaDB.similarity_search_with_relevance_scores(k=3)
     ▼
 Results:
   Doc1 (score: 0.89): "Acceleration is the rate of change..."
   Doc2 (score: 0.72): "When a force acts on a body..."
   Doc3 (score: 0.21): "The history of thermodynamics..." ← FILTERED (< 0.3)
     │
     │ Filter: score >= 0.3
     ▼
 Relevant Chunks: [Doc1, Doc2]
     │
     │ Return as JSON to LangGraph
     ▼
 LLM synthesizes answer using retrieved chunks as primary source
```

---

## 9. LangGraph — Chatbot State Machine

### Graph Structure

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
              ┌────│ agent_node  │────┐
              │    └─────────────┘    │
              │                       │
         has tool_calls          no tool_calls
              │                       │
              ▼                       ▼
       ┌─────────────┐        ┌─────────────┐
       │  ToolNode   │        │     END     │
       └──────┬──────┘        └─────────────┘
              │
              │ (tool results added to state)
              │
              └──────→ agent_node (synthesize)
                              │
                         no tool_calls
                              │
                              ▼
                       ┌─────────────┐
                       │     END     │
                       └─────────────┘
```

### State Accumulation

LangGraph uses `add_messages` to accumulate messages across node executions:

```
After agent_node (1st pass):
  [SystemMessage, HumanMessage, AIMessage(tool_calls)]

After ToolNode:
  [SystemMessage, HumanMessage, AIMessage(tool_calls), ToolMessage(results)]

After agent_node (2nd pass - synthesis):
  [SystemMessage, HumanMessage, AIMessage(tool_calls), ToolMessage(results), AIMessage(final answer)]
```

### Memory (MemorySaver)

- Each conversation has a `thread_id`
- `MemorySaver` stores conversation state in memory
- Same `thread_id` = continued conversation with full context
- Different `thread_id` = fresh conversation

---

## 10. Streaming (SSE) Flow

The frontend uses Server-Sent Events for real-time token streaming:

```
Frontend (Streamlit)                    Backend (FastAPI)
     │                                       │
     │ POST /chat/stream {query, thread_id}  │
     │ ─────────────────────────────────────→ │
     │                                       │
     │     data: {"type":"tool_start",       │ ← on_tool_start event
     │ ←──────── "tool":"retrieve_from..."}  │
     │                                       │
     │     data: {"type":"source_docs",      │ ← on_tool_end event (RAG only)
     │ ←──────── "docs":["chunk1","chunk2"]} │
     │                                       │
     │     data: {"type":"content",          │ ← on_chat_model_stream event
     │ ←──────── "text":"Based on"}          │   (tagged: stream_response)
     │                                       │
     │     data: {"type":"content",          │
     │ ←──────── "text":" the"}              │
     │                                       │
     │     data: {"type":"content",          │
     │ ←──────── "text":" stored"}           │
     │                                       │
     │     ...more tokens...                 │
     │                                       │
     │     data: {"type":"done"}             │ ← stream complete
     │ ←─────────────────────────────────── │
     │                                       │
```

The streaming endpoint also captures `on_chain_end` events for nodes named `greeting_node` or `evaluator_node` (reserved for future graph expansions).

---

## 11. Legacy MCP Servers (server/)

> These files are **not used by the production API** but remain functional for standalone testing and as the MCP reference implementation.

### 📄 `server/rag_server.py` — FastMCP RAG Tool Server

- **Framework**: FastMCP (`mcp = FastMCP("rag_knowledge_server")`)
- **Transport**: stdio (spawned as subprocess)
- **Tool**: `retrieve_from_knowledge_base(query: str) -> str`
- **Behavior**: Same logic as `client/rag_tool.py` but runs in a separate process. Initializes embedding model + ChromaDB on every call (no lazy singleton).
- **Run**: `python server/rag_server.py`

### 📄 `server/math_server.py` — FastMCP Calculator Server

- **Framework**: FastMCP (`mcp = FastMCP("math_tools_server")`)
- **Transport**: stdio (spawned as subprocess)
- **Tools**: `calculator_add(a, b)`, `calculator_multiply(a, b)`
- **Run**: `python server/math_server.py`

### When Legacy MCP Path Is Used

```
client/chatbot.py:run_interactive()
  → client/mcp_client.py:get_mcp_tools()
    → spawns: server/rag_server.py (stdio subprocess)
    → spawns: server/math_server.py (stdio subprocess)
    → returns tools discovered via MCP protocol
  → create_chatbot(mcp_tools)
```

---

## 12. Environment Variables

| Variable | Required | Default | Used By | Purpose |
|----------|----------|---------|---------|---------|
| `GROQ_API_KEY` | ✅ Yes | — | `chatbot.py` | Groq API authentication |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | `chatbot.py` | Which Groq model to use |
| `GOOGLE_API_KEY` | ✅ Yes | — | `rag_tool.py`, `rag_server.py`, `embed_documents.py` | Google Gemini API key |
| `EMBEDDING_MODEL` | No | `models/gemini-embedding-2` | `rag_tool.py`, `rag_server.py`, `embed_documents.py` | Gemini embedding model |
| `CHROMA_DIR` | No | `./chroma_db` | `rag_tool.py`, `rag_server.py`, `embed_documents.py` | ChromaDB storage path |
| `COLLECTION_NAME` | No | `rag_knowledge_base` | `rag_tool.py`, `rag_server.py`, `embed_documents.py` | ChromaDB collection name |
| `API_HOST` | No | `0.0.0.0` | `api/server.py` | FastAPI bind address |
| `API_PORT` | No | `8000` | `api/server.py`, `start.py` | FastAPI port |
| `API_URL` | No | `http://127.0.0.1:8000` | `frontend/app.py` | Backend URL for frontend |
| `PORT` | No (Render sets it) | — | `start.py` | Render injects this for the web service |

---

## 13. Deployment on Render

### Why Render?

| Platform | Works? | Persistent Processes? | Free RAM | Verdict |
|----------|--------|----------------------|----------|--------|
| **Render** | ✅ Yes | ✅ Web Service | 512MB | ✅ Best fit |
| Vercel | ❌ No | ❌ Serverless only | 250MB | ❌ Won't work |
| Railway | ✅ Yes | ✅ | 512MB | ✅ Also works |
| Fly.io | ✅ Yes | ✅ | 256MB | ⚠️ Tight RAM |

### Deployment Steps

```bash
# 1. Push your code to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USER/mcp-rag-chatbot.git
git push -u origin main

# 2. Go to https://dashboard.render.com
# 3. New → Web Service → Connect your GitHub repo
# 4. Render auto-detects render.yaml and configures:
#    Build:  pip install -r requirements.txt
#    Start:  python start.py
# 5. Add environment variables in Render dashboard:
#    GROQ_API_KEY     = gsk_your_key
#    GOOGLE_API_KEY   = your_google_key
```

### What Happens on Cold Start

```
start.py runs:
  1. run_embedding() checks if chroma_db/chroma.sqlite3 exists
  2. If missing → runs embed_documents.py as subprocess (~30s)
     - Reads storage/*.txt
     - Embeds via Google Gemini API (batches of 80, 65s sleep)
     - Creates chroma_db/
  3. start_server() → os.execvp uvicorn on PORT
  4. FastAPI lifespan → initialize_tools() loads direct tools
  5. create_chatbot() builds LangGraph
  6. API is ready to serve requests
```

### Files for Deployment

| File | Purpose |
|------|--------|
| `render.yaml` | Render deployment config (build/start commands, env vars) |
| `start.py` | Startup script (check embedding + launch uvicorn) |
| `Dockerfile` | Docker container definition |
| `docker-compose.yml` | Docker Compose orchestration |
| `requirements.txt` | Dependencies (no PyTorch = fits in 512MB RAM) |

---

## 14. How to Run — Commands Reference

### First-Time Setup
```bash
# 1. Navigate to project
cd mcp_rag_chatbot

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env → add your GROQ_API_KEY and GOOGLE_API_KEY

# 5. Embed documents (one-time)
python embed_documents.py
```

### Running the Application
```bash
# Terminal 1: Start API server
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run frontend/app.py

# OR: Test via CLI (uses legacy MCP subprocess path)
python -m client.chatbot
```

### Adding New Documents
```bash
# 1. Place .txt files in storage/
# 2. Re-run embedding
python embed_documents.py
```

---

## 15. Design Decisions & Trade-offs

### 1. Direct Tools vs MCP Subprocess Servers

**Old approach** (`server/rag_server.py` + `client/mcp_client.py`): Tools ran as FastMCP stdio subprocesses, discovered via the MCP protocol.

**New approach** (`client/rag_tool.py` + `client/math_tools.py`): Tools are direct LangChain `@tool` functions running in-process.

**Why we switched**:
- ✅ Eliminates stdio subprocess failures on cloud platforms (Render, etc.)
- ✅ No subprocess management or lifecycle issues
- ✅ Faster tool invocation (no IPC overhead)
- ✅ Lazy singleton for vector store (initialized once, reused)
- ❌ Tools now coupled to the API process (can't run as independent services)

> **Legacy MCP path preserved**: `server/` and `client/mcp_client.py` still work for `python -m client.chatbot` CLI testing and as a reference implementation.

### 2. RAG Tool Returns Raw Chunks (Not Synthesized Answers)

**Old approach** (tempcodes/retrieve.py): The RAG tool called an LLM internally to synthesize an answer.

**New approach**: The tool returns raw document chunks. The LangGraph chatbot's LLM synthesizes the answer.

**Why**: 
- Tools should be **pure data functions** — fetch data, return it
- The LLM that interacts with the user should do the synthesis — it has the conversation context
- Avoids double LLM calls (one in tool + one in chatbot)
- Makes tools reusable by any client

### 3. Google Gemini Embeddings vs HuggingFace Local

**Choice**: Google Gemini `gemini-embedding-2` (768 dims, cloud API) over HuggingFace `all-MiniLM-L6-v2` (384 dims, local)

**Why we switched from HuggingFace**:
- HuggingFace requires `torch` (PyTorch) = ~2GB install, ~700MB+ RAM
- Free hosting tiers (Render 512MB, Railway 512MB) can't handle it
- Google Gemini `gemini-embedding-2` is:
  - ✅ Free tier: 1500 requests/day
  - ✅ Lightweight: `langchain-google-genai` ~10MB (vs PyTorch ~2GB)
  - ✅ Better quality: 768 dims vs 384 dims
  - ❌ Needs API key (free at aistudio.google.com/apikey)
  - ❌ Needs internet (cloud API call)

### 4. Lazy Singleton Vector Store (rag_tool.py)

**Design**: `_get_vector_store()` initializes ChromaDB + embedding model on first call, caches globally.

**Why**:
- Avoids re-initializing the embedding model on every tool call
- First request is slower (~2-3s), all subsequent requests are instant
- Contrast with `server/rag_server.py` which reinitializes on every call

### 5. SSE Streaming vs Polling

**Choice**: Server-Sent Events (SSE) via `POST /chat/stream`

**Why**:
- Real-time token-by-token display (no waiting for full response)
- Lightweight (one HTTP connection, text/event-stream)
- Native browser support
- Shows tool execution status in real-time

### 6. MemorySaver vs External Persistence

**Choice**: In-memory `MemorySaver` for conversation state

**Trade-off**:
- ✅ Zero setup, instant
- ❌ Lost when server restarts
- **Future**: Swap to `SqliteSaver` or `PostgresSaver` for persistence

### 7. Background Tool Initialization

**Design**: `initialize_tools()` runs as `asyncio.create_task()` inside the lifespan handler.

**Why**:
- Uvicorn binds to the port immediately (health checks pass right away)
- Tool loading happens in the background
- `/health` returns `{"status": "starting up"}` until ready
- Other endpoints return 503 until `is_ready = True`
