# 🏗️ ARCHITECTURE — Multi-Server MCP RAG Chatbot

> Complete system design, request flows, component documentation, and tech stack reference.
> Use this document to understand **what each piece does**, **how data flows**, and **how the system is wired together**.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Tech Stack Deep-Dive](#2-tech-stack-deep-dive)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Request Flow — Step by Step](#4-request-flow--step-by-step)
5. [Component Documentation](#5-component-documentation)
6. [MCP Protocol — How It Works](#6-mcp-protocol--how-it-works)
7. [RAG Pipeline — Embedding to Answer](#7-rag-pipeline--embedding-to-answer)
8. [Data Flow Diagram](#8-data-flow-diagram)
9. [LangGraph — Chatbot State Machine](#9-langgraph--chatbot-state-machine)
10. [Environment Variables](#10-environment-variables)
11. [Deployment on Render](#11-deployment-on-render)
12. [How to Run — Commands Reference](#12-how-to-run--commands-reference)
13. [Design Decisions & Trade-offs](#13-design-decisions--trade-offs)

---

## 1. System Overview

This project is a **multi-server chatbot** that:

1. **Receives user questions** via a Streamlit chat UI (or REST API)
2. **Routes to an LLM** (Groq — Llama 3.3 70B) which decides if a tool is needed
3. **Calls MCP tool servers** if needed (RAG knowledge retrieval or calculator)
4. **Retrieves relevant documents** from a ChromaDB vector store
5. **Synthesizes an answer** using the retrieved context
6. **Returns "no data available"** if the query topic isn't in the knowledge base

### What makes this different from a basic chatbot?

| Feature | Basic Chatbot | This Project |
|---------|--------------|-------------|
| Knowledge | General LLM knowledge only | Grounded in YOUR documents |
| Tools | None | MCP tool servers (extensible) |
| Memory | Stateless | Thread-based conversation memory |
| Architecture | Monolithic | Modular (server/client/API/frontend) |
| Tool Calling | Manual if/else | LLM decides automatically via LangGraph |

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
| **FastMCP** | MCP tool server framework | Simple decorator-based tool definition |
| **langchain-mcp-adapters** | MCP client bridge | Connects LangChain tools to MCP servers |

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

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                              │
│  ┌──────────────────┐     ┌──────────────────────────────────────┐  │
│  │  Streamlit UI     │     │  Any HTTP Client (curl, Postman)    │  │
│  │  frontend/app.py  │     │  Direct API access                  │  │
│  └────────┬─────────┘     └─────────────────┬────────────────────┘  │
│           │ HTTP POST /chat                  │                      │
└───────────┼──────────────────────────────────┼──────────────────────┘
            │                                  │
            ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API LAYER (FastAPI)                           │
│  api/server.py                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Endpoints:                                                     │  │
│  │   POST /chat     → Invoke chatbot, return response            │  │
│  │   GET  /health   → Server status + connected tools            │  │
│  │   GET  /tools    → List available MCP tools                   │  │
│  │                                                                │  │
│  │ Lifespan:                                                      │  │
│  │   startup  → Connect MCP servers, create chatbot              │  │
│  │   shutdown → Cleanup connections                              │  │
│  └─────────────────────────────┬──────────────────────────────────┘  │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER (LangGraph)                   │
│  client/chatbot.py                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │   START → [chat_node] → {has tool calls?}                     │  │
│  │                             │            │                     │  │
│  │                            YES          NO                     │  │
│  │                             │            │                     │  │
│  │                             ▼            ▼                     │  │
│  │                       [tool_node]      [END]                   │  │
│  │                             │                                  │  │
│  │                             └──→ [chat_node] → [END]          │  │
│  │                              (synthesize with tool results)    │  │
│  │                                                                │  │
│  │   MemorySaver: Thread-based conversation persistence          │  │
│  └─────────────────────────────┬──────────────────────────────────┘  │
│                                │                                    │
│  client/mcp_client.py          │ tool invocation                    │
│  ┌─────────────────────────────┼──────────────────────────────────┐  │
│  │ MultiServerMCPClient        ▼                                  │  │
│  │   Connects to MCP servers via stdio subprocess                │  │
│  │   Discovers tools from all servers                             │  │
│  │   Returns tools for LLM binding                               │  │
│  └──────────┬────────────────────────────────┬────────────────────┘  │
└─────────────┼────────────────────────────────┼──────────────────────┘
              │ stdio                           │ stdio
              ▼                                 ▼
┌──────────────────────────┐   ┌──────────────────────────────────────┐
│  MCP SERVER 1            │   │  MCP SERVER 2                        │
│  server/math_server.py   │   │  server/rag_server.py                │
│                          │   │                                      │
│  Tools:                  │   │  Tools:                              │
│  • calculator_add(a, b)  │   │  • retrieve_from_knowledge_base(q)  │
│  • calculator_multiply   │   │                                      │
│    (a, b)                │   │  ┌────────────────────────────┐      │
│                          │   │  │  Google Gemini Embeddings  │      │
│  Pure functions,         │   │  │  Embed query → vector      │      │
│  no external deps        │   │  └─────────────┬──────────────┘      │
│                          │   │                │                      │
│                          │   │                ▼                      │
│                          │   │  ┌────────────────────────────┐      │
│                          │   │  │  ChromaDB Vector Store     │      │
│                          │   │  │  Similarity search (k=3)   │      │
│                          │   │  │  Score threshold > 0.3     │      │
│                          │   │  └────────────────────────────┘      │
└──────────────────────────┘   └──────────────────────────────────────┘
```

---

## 4. Request Flow — Step by Step

### Scenario: User asks *"What is Newton's second law?"*

```
Step 1  │ USER types message in Streamlit UI
        │ frontend/app.py sends POST /chat to FastAPI
        │
Step 2  │ FastAPI receives ChatRequest{query, thread_id}
        │ api/server.py invokes LangGraph chatbot
        │
Step 3  │ LangGraph: chat_node runs
        │ Groq LLM sees system prompt + user message + tool schemas
        │ LLM decides: "I should search the knowledge base"
        │ Returns AIMessage with tool_calls:
        │   [{name: "retrieve_from_knowledge_base", args: {query: "Newton's second law"}}]
        │
Step 4  │ LangGraph: should_call_tool → "tool_node"
        │ ToolNode executes the MCP tool via stdio
        │
Step 5  │ MCP Client sends tool call to rag_server.py subprocess
        │ rag_server.py:
        │   a) Embeds query with Google Gemini API
        │   b) Searches ChromaDB for similar vectors (k=3)
        │   c) Filters by relevance score (> 0.3)
        │   d) Returns JSON: {found: true, documents: ["...", "..."], scores: [0.87, 0.72]}
        │
Step 6  │ LangGraph: tool_node → chat_node (loop back)
        │ ToolMessage with retrieval results added to conversation
        │
Step 7  │ LangGraph: chat_node runs AGAIN
        │ Groq LLM now sees: system prompt + user Q + tool results
        │ System prompt says: "Use ONLY the returned documents"
        │ LLM synthesizes answer from the retrieved chunks
        │ Returns AIMessage with content (final answer)
        │
Step 8  │ LangGraph: should_call_tool → END (no more tool calls)
        │
Step 9  │ FastAPI extracts: response text, tools_used, source_docs
        │ Returns ChatResponse to Streamlit
        │
Step 10 │ Streamlit displays:
        │   - AI response text
        │   - 🔧 Tool badge: "retrieve_from_knowledge_base"
        │   - 📄 Expandable source documents
```

### Scenario: User asks *"What is quantum entanglement?"* (NOT in knowledge base)

```
Steps 1-5 │ Same as above...
           │
Step 5     │ rag_server.py:
           │   Searches ChromaDB → all scores < 0.3 threshold
           │   Returns: {found: false, documents: [], message: "No relevant documents..."}
           │
Steps 6-7  │ chat_node sees {found: false}
           │ System prompt says: "If found is false, tell user data is not available"
           │ LLM responds: "This information is not available in the stored documents."
           │
Steps 8-10 │ Response delivered to user
```

### Scenario: User asks *"What is 25 + 37?"*

```
Step 3  │ Groq LLM sees the math question
        │ Decides to use calculator_add tool
        │ Returns: tool_calls: [{name: "calculator_add", args: {a: 25, b: 37}}]
        │
Step 5  │ MCP Client calls math_server.py
        │ math_server.py: returns 62
        │
Step 7  │ LLM synthesizes: "25 + 37 = 62"
```

### Scenario: User says *"Hello, how are you?"*

```
Step 3  │ Groq LLM sees a greeting
        │ No tool call needed — responds directly
        │ Returns: AIMessage with content "Hello! I'm doing great..."
        │
Step 4  │ should_call_tool → END (no tool calls)
        │ Skips tool_node entirely
```

---

## 5. Component Documentation

### 📄 `client/config.py` — Configuration Hub

**Purpose**: Single source of truth for all environment variables.

| Function | What It Does |
|----------|-------------|
| `validate_config()` | Checks all required env vars are set. Exits with clear error if not. |
| `print_config()` | Debug output with masked API keys. |

| Constant | Source | Default |
|----------|--------|---------|
| `GROQ_API_KEY` | `.env` | (required) |
| `GROQ_MODEL` | `.env` | `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL` | `.env` | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHROMA_DIR` | `.env` | `./chroma_db` |
| `COLLECTION_NAME` | `.env` | `rag_knowledge_base` |
| `PROJECT_ROOT` | computed | Parent of `client/` dir |

---

### 📄 `server/rag_server.py` — RAG Knowledge Retrieval Tool

**Purpose**: MCP tool server that searches ChromaDB for documents matching a query.

| Function | Inputs | Outputs | What It Does |
|----------|--------|---------|-------------|
| `retrieve_from_knowledge_base(query)` | `query: str` | `JSON str` | 1. Embeds query via Google Gemini API<br>2. Searches ChromaDB (k=3)<br>3. Filters by score > 0.3<br>4. Returns `{found, documents, scores}` |

**Key Design Decision**: This tool does **NOT** call an LLM internally. It returns raw document chunks. The LLM synthesis happens in `chatbot.py`. This separation keeps tools pure and reusable.

**Return Schema**:
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
  "message": "No relevant documents found..."
}
```

---

### 📄 `server/math_server.py` — Calculator Tool

**Purpose**: Demo MCP tool server for basic arithmetic.

| Function | Inputs | Outputs |
|----------|--------|---------|
| `calculator_add(a, b)` | `a: int, b: int` | `int` (sum) |
| `calculator_multiply(a, b)` | `a: int, b: int` | `int` (product) |

---

### 📄 `client/mcp_client.py` — Multi-Server MCP Client

**Purpose**: Connects to all MCP servers and returns tools ready for LLM binding.

| Function | What It Does | Returns |
|----------|-------------|---------|
| `get_mcp_server_config()` | Builds server config dict with paths | `dict` |
| `get_mcp_tools()` | Connects via stdio, discovers tools | `(client, tools, named_tools)` |

**How servers are connected**:
```
mcp_client.py
    │
    ├── stdio subprocess → python server/rag_server.py
    │                       └── Exposes: retrieve_from_knowledge_base
    │
    └── stdio subprocess → python server/math_server.py
                            └── Exposes: calculator_add, calculator_multiply
```

> **Important**: The `client` object must stay alive! If it's garbage collected, the stdio subprocesses die and tools stop working.

---

### 📄 `client/chatbot.py` — LangGraph Chatbot Engine

**Purpose**: The brain — a stateful chatbot that decides when to use tools.

| Function | What It Does |
|----------|-------------|
| `create_chatbot(tools)` | Builds and compiles the LangGraph state graph |
| `run_interactive()` | CLI testing mode (no API/frontend needed) |

**Internal Nodes**:

| Node | Role | Input | Output |
|------|------|-------|--------|
| `chat_node` | Invokes Groq LLM with tools bound | `ChatState` (messages) | `AIMessage` (may contain tool_calls) |
| `tool_node` | Executes MCP tool calls | `AIMessage.tool_calls` | `ToolMessage` (tool results) |

**Routing Logic** (`should_call_tool`):
```python
if last_message has tool_calls → "tool_node"
else → END
```

**State Schema**:
```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # add_messages annotation = LangGraph auto-appends new messages
```

---

### 📄 `api/server.py` — FastAPI Backend

**Purpose**: HTTP API wrapping the chatbot for the frontend.

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/health` | GET | — | `{status, tools_connected, tool_names}` |
| `/tools` | GET | — | `{tools: [{name, description}]}` |
| `/chat` | POST | `{query, thread_id}` | `{response, tools_used, source_docs, thread_id}` |

**Lifespan Management**:
- On startup: connects to MCP servers, creates chatbot
- On shutdown: cleans up connections
- MCP connections persist for the lifetime of the API process

---

### 📄 `frontend/app.py` — Streamlit Chat UI

**Purpose**: User-facing chat interface.

| Feature | Implementation |
|---------|---------------|
| Chat input | `st.chat_input()` |
| Message history | `st.session_state["message_history"]` |
| Tool badges | Shown below AI responses when tools are used |
| Source docs | Expandable `st.expander` with retrieved chunks |
| API health | Sidebar indicator with connected tools |
| Suggested queries | Sidebar with example questions |

---

### 📄 `embed_documents.py` — Document Embedding Pipeline

**Purpose**: One-time script to process raw text files into ChromaDB vectors.

| Step | Function | What Happens |
|------|----------|-------------|
| 1 | `load_text_file(path)` | Reads `.txt` file content |
| 2 | `split_text(text)` | Chunks into 1000-char pieces with 50-char overlap |
| 3 | `embed_and_store(chunks)` | HuggingFace embeds → ChromaDB stores |

**Chunking Strategy**:
```
RecursiveCharacterTextSplitter:
  chunk_size = 1000 chars
  chunk_overlap = 50 chars
  separators = ["\n\n", "\n", ". ", ", ", " "]
  
  Tries paragraph splits first, then sentences, then words.
  Overlap ensures context isn't lost at chunk boundaries.
```

---

## 6. MCP Protocol — How It Works

**MCP (Model Context Protocol)** is a standard for connecting LLM applications to external tools and data sources.

### Server Side (Tool Provider)
```python
# server/rag_server.py
from fastmcp import FastMCP

mcp = FastMCP("rag_knowledge_server")

@mcp.tool()
def retrieve_from_knowledge_base(query: str) -> str:
    # Tool implementation
    ...

mcp.run()  # Starts stdio server
```

### Client Side (Tool Consumer)
```python
# client/mcp_client.py
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "rag_knowledge_server": {
        "transport": "stdio",
        "command": "python",
        "args": ["server/rag_server.py"]
    }
})

tools = await client.get_tools()  # Discovers tools from server
llm_with_tools = llm.bind_tools(tools)  # LLM can now call tools
```

### Communication Flow
```
Client                          Server (subprocess)
  │                                │
  │ ──── spawn process ──────────→ │ (python rag_server.py)
  │                                │
  │ ──── list_tools request ─────→ │
  │ ←─── tool schemas ───────────  │
  │                                │
  │ ──── call_tool request ──────→ │ (with args)
  │ ←─── tool result ────────────  │
  │                                │
  │ ──── (more calls...) ───────→  │
  │                                │
```

**Transport: stdio** means the client spawns the server as a subprocess and communicates via stdin/stdout. No HTTP, no ports — just process I/O.

---

## 7. RAG Pipeline — Embedding to Answer

### Phase 1: Document Embedding (One-Time)

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
     │ Chroma.from_texts()             │
     └─────────────────────────────────┘
```

### Phase 2: Query & Retrieval (Per User Query)

```
User Query: "What is acceleration?"
     │
     │ HuggingFace Embeddings
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
     │ Return as JSON to LLM
     ▼
 LLM synthesizes answer using ONLY these chunks
```

---

## 8. Data Flow Diagram

Complete request → response data transformation:

```
┌──────────────────────────────────────────────────────────────────────┐
│                      COMPLETE DATA FLOW                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Input                                                          │
│  "What forces act on a moving body?"                                │
│       │                                                              │
│       ▼                                                              │
│  HTTP Request                                                        │
│  POST /chat                                                          │
│  {"query": "What forces act...", "thread_id": "session-1"}          │
│       │                                                              │
│       ▼                                                              │
│  LangGraph State                                                     │
│  {messages: [SystemMessage, HumanMessage("What forces...")]}        │
│       │                                                              │
│       ▼                                                              │
│  Groq LLM Response (chat_node)                                      │
│  AIMessage(tool_calls=[{                                            │
│    name: "retrieve_from_knowledge_base",                            │
│    args: {query: "forces acting on moving body"},                   │
│    id: "call_abc123"                                                │
│  }])                                                                 │
│       │                                                              │
│       ▼                                                              │
│  MCP Tool Execution (tool_node → rag_server.py)                     │
│  Input:  query="forces acting on moving body"                       │
│  Output: '{"found":true,"documents":["...","..."],"scores":[.87]}'  │
│       │                                                              │
│       ▼                                                              │
│  ToolMessage added to state                                          │
│  {messages: [System, Human, AI(tool_call), ToolMessage(results)]}   │
│       │                                                              │
│       ▼                                                              │
│  Groq LLM Synthesis (chat_node, second pass)                        │
│  AIMessage(content="Based on the stored documents, the forces...")   │
│       │                                                              │
│       ▼                                                              │
│  HTTP Response                                                       │
│  {                                                                   │
│    "response": "Based on the stored documents...",                  │
│    "tools_used": ["retrieve_from_knowledge_base"],                  │
│    "source_docs": ["Forces that act on...", "..."],                 │
│    "thread_id": "session-1"                                         │
│  }                                                                   │
│       │                                                              │
│       ▼                                                              │
│  Streamlit UI                                                        │
│  ┌─────────────────────────────────────┐                            │
│  │ 🤖 Based on the stored documents,  │                            │
│  │    the forces acting on a moving    │                            │
│  │    body include...                  │                            │
│  │                                     │                            │
│  │ 🔧 retrieve_from_knowledge_base    │                            │
│  │ 📄 Source Documents ▼              │                            │
│  └─────────────────────────────────────┘                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
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
              ┌────│  chat_node  │────┐
              │    └─────────────┘    │
              │                       │
         has tool_calls          no tool_calls
              │                       │
              ▼                       ▼
       ┌─────────────┐        ┌─────────────┐
       │  tool_node  │        │     END     │
       └──────┬──────┘        └─────────────┘
              │
              │ (tool results added to state)
              │
              └──────→ chat_node (synthesize)
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
After chat_node (1st pass):
  [SystemMessage, HumanMessage, AIMessage(tool_calls)]

After tool_node:
  [SystemMessage, HumanMessage, AIMessage(tool_calls), ToolMessage(results)]

After chat_node (2nd pass - synthesis):
  [SystemMessage, HumanMessage, AIMessage(tool_calls), ToolMessage(results), AIMessage(final answer)]
```

### Memory (MemorySaver)

- Each conversation has a `thread_id`
- `MemorySaver` stores conversation state in memory
- Same `thread_id` = continued conversation with full context
- Different `thread_id` = fresh conversation

---

## 10. Environment Variables

| Variable | Required | Default | Used By | Purpose |
|----------|----------|---------|---------|---------|
| `GROQ_API_KEY` | ✅ Yes | — | `chatbot.py` | Groq API authentication |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | `chatbot.py` | Which Groq model to use |
| `GOOGLE_API_KEY` | ✅ Yes | — | `rag_server.py`, `embed_documents.py` | Google Gemini API key (get at aistudio.google.com/apikey) |
| `EMBEDDING_MODEL` | No | `models/gemini-embedding-2` | `rag_server.py`, `embed_documents.py` | Google Gemini embedding model |
| `CHROMA_DIR` | No | `./chroma_db` | `rag_server.py`, `embed_documents.py` | ChromaDB storage path |
| `COLLECTION_NAME` | No | `rag_knowledge_base` | `rag_server.py`, `embed_documents.py` | ChromaDB collection name |
| `API_HOST` | No | `0.0.0.0` | `api/server.py` | FastAPI bind address |
| `API_PORT` | No | `8000` | `api/server.py` | FastAPI port |
| `PORT` | No (Render sets it) | — | `start.py` | Render injects this for the web service |

---

## 11. Deployment on Render

### Why Render?

| Platform | stdio Works? | Persistent Processes? | Free RAM | Verdict |
|----------|-------------|----------------------|----------|--------|
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
  1. Checks if chroma_db/ exists (ephemeral disk)
  2. If missing → runs embed_documents.py (~30s)
     - Reads storage/*.txt
     - Embeds via Google Gemini API
     - Creates chroma_db/
  3. Starts uvicorn on PORT (set by Render)
  4. FastAPI lifespan connects MCP servers via stdio
  5. API is ready to serve requests
```

### stdio on Render — How It Works

Render Web Services are **real servers** (not serverless). Your FastAPI process spawns MCP server subprocesses via stdio — this works exactly like local development. No transport changes needed.

```
Render Web Service (single process)
└── python start.py
    └── uvicorn api.server:app
        └── FastAPI lifespan:
            ├── subprocess: python server/rag_server.py  (stdio)
            └── subprocess: python server/math_server.py (stdio)
```

### Files for Deployment

| File | Purpose |
|------|--------|
| `render.yaml` | Render deployment config (build/start commands, env vars) |
| `start.py` | Startup script (re-embed + launch uvicorn) |
| `requirements.txt` | Dependencies (no PyTorch = fits in 512MB RAM) |

---

## 12. How to Run — Commands Reference

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
# Edit .env → add your GROQ_API_KEY

# 5. Embed documents (one-time)
python embed_documents.py
```

### Running the Application
```bash
# Terminal 1: Start API server
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run frontend/app.py

# OR: Test via CLI (no server needed)
python -m client.chatbot
```

### Adding New Documents
```bash
# 1. Place .txt files in storage/
# 2. Re-run embedding
python embed_documents.py
```

---

## 13. Design Decisions & Trade-offs

### 1. RAG Tool Returns Raw Chunks (Not Synthesized Answers)

**Old approach** (tempcodes/retrieve.py): The RAG tool called an LLM internally to synthesize an answer from retrieved docs.

**New approach**: The tool returns raw document chunks. The LangGraph chatbot's LLM synthesizes the answer.

**Why**: 
- Tools should be **pure data functions** — fetch data, return it
- The LLM that interacts with the user should do the synthesis — it has the conversation context
- Avoids double LLM calls (one in tool + one in chatbot)
- Makes tools reusable by any client

### 2. Google Gemini Embeddings vs HuggingFace Local

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

### 3. stdio Transport vs HTTP Transport for MCP

**Choice**: stdio (subprocess) for local servers

**Why**: 
- No port management needed
- No network overhead
- Server lifecycle managed by client
- Simple for local development

**When to use HTTP**: When servers are remote, deployed separately, or shared across clients.

### 4. MemorySaver vs External Persistence

**Choice**: In-memory `MemorySaver` for conversation state

**Trade-off**:
- ✅ Zero setup, instant
- ❌ Lost when server restarts
- **Future**: Swap to `SqliteSaver` or `PostgresSaver` for persistence
