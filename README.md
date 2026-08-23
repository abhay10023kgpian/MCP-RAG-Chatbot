# 🤖 Multi-Server MCP RAG Chatbot

A production-ready chatbot that uses **Model Context Protocol (MCP)** to connect to multiple tool servers, including a **RAG (Retrieval-Augmented Generation)** pipeline backed by **ChromaDB**. Powered by **Groq** (Llama 3.3 70B) and orchestrated with **LangGraph**.

## ⚡ Quick Start

### 1. Install Dependencies
```bash
cd mcp_rag_chatbot
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy the template and add your API keys
cp .env.example .env
# Edit .env:
#   GROQ_API_KEY=gsk_your_key_here
#   GOOGLE_API_KEY=your_google_key_here  (get at https://aistudio.google.com/apikey)
```

### 3. Embed Documents (One-Time)
```bash
python embed_documents.py
```
This processes files in `storage/` and creates the ChromaDB vector store.

### 4. Run the API Server
```bash
# From the mcp_rag_chatbot directory:
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### 5. Run the Frontend
```bash
# In a separate terminal:
streamlit run frontend/app.py
```

### 6. (Optional) Test CLI Mode
```bash
python -m client.chatbot
```

## 🚀 Deploy on Render

```bash
# Push to GitHub, then connect to Render:
# Build Command:  pip install -r requirements.txt
# Start Command:  python start.py
# Set env vars:   GROQ_API_KEY, GOOGLE_API_KEY
```

See [ARCHITECTURE.md → Deployment on Render](./ARCHITECTURE.md#11-deployment-on-render) for full steps.

## 🏗️ Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full system design, request flow diagrams, and component documentation.

## 📂 Project Structure

```
mcp_rag_chatbot/
├── .env                    # Your API keys (git-ignored)
├── .env.example            # Template
├── requirements.txt        # Dependencies
├── ARCHITECTURE.md         # ★ Full system documentation
├── README.md               # This file
├── start.py                # Production startup (Render)
├── render.yaml             # Render deployment config
│
├── storage/                # Raw documents for RAG
│   └── *.txt files
│
├── embed_documents.py      # One-time embedding script
├── chroma_db/              # Generated vector store
│
├── server/                 # MCP Tool Servers
│   ├── rag_server.py       # Knowledge base retrieval
│   └── math_server.py      # Calculator (demo)
│
├── client/                 # LLM Orchestrator
│   ├── config.py           # Centralized config
│   ├── mcp_client.py       # Multi-server client
│   └── chatbot.py          # LangGraph chatbot
│
├── api/                    # REST API
│   └── server.py           # FastAPI backend
│
└── frontend/               # UI
    └── app.py              # Streamlit chat
```

## 🔑 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Groq (Llama 3.3 70B) | Chat intelligence |
| Embeddings | Google Gemini (gemini-embedding-2) | Document vectorization |
| Vector DB | ChromaDB | Similarity search |
| Orchestration | LangGraph | Stateful chatbot flow |
| Tool Protocol | MCP (FastMCP) | Server-client tool communication |
| API | FastAPI | REST backend |
| Frontend | Streamlit | Chat interface |
| Deployment | Render | Cloud hosting |
