"""
server.py — FastAPI REST API Backend
======================================
Exposes the LangGraph chatbot via HTTP endpoints.

Endpoints:
  GET  /health     — Health check
  POST /chat       — Send message, get response
  GET  /tools      — List available MCP tools

Based on: Langraph_tutorials/chatbot_server.py
Key changes: 
  - Async support for MCP tool calls
  - CORS enabled for frontend
  - Returns tool usage metadata alongside responses
  - Lifespan-managed MCP connections

Usage:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# ─── Load .env from project root ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

from client.mcp_client import get_mcp_tools
from client.chatbot import create_chatbot


# ─── Application State (populated on startup) ───
app_state = {
    "chatbot": None,
    "make_config": None,
    "mcp_client": None,
    "tools": None,
    "named_tools": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler — connects to MCP servers on startup,
    cleans up on shutdown.
    
    Flow:
      startup:
        1. Connect to all MCP servers
        2. Discover available tools
        3. Create LangGraph chatbot with tools bound
      
      shutdown:
        1. Close MCP client connections
    """
    print("\n🚀 Starting MCP RAG Chatbot API...")

    # ── Connect to MCP servers ──
    print("🔌 Connecting to MCP servers...")
    client, tools, named_tools = await get_mcp_tools()
    app_state["mcp_client"] = client
    app_state["tools"] = tools
    app_state["named_tools"] = named_tools

    # ── Create chatbot ──
    print("🤖 Creating LangGraph chatbot...")
    chatbot, make_config = await create_chatbot(tools)
    app_state["chatbot"] = chatbot
    app_state["make_config"] = make_config

    print("✅ API ready!\n")
    yield

    # ── Cleanup ──
    print("\n🛑 Shutting down...")


# ─── FastAPI App ───
app = FastAPI(
    title="MCP RAG Chatbot API",
    description="Multi-Server MCP Chatbot with RAG retrieval from ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow Streamlit frontend) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Schemas ───
class ChatRequest(BaseModel):
    """
    Incoming chat request.
    
    Attributes:
        query: The user's message
        thread_id: Conversation thread ID for memory persistence.
                   Same thread_id = same conversation context.
    """
    query: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    """
    Outgoing chat response.
    
    Attributes:
        response: The AI assistant's text response
        tools_used: List of tool names that were called during this turn
        source_docs: Retrieved document snippets (if RAG was used)
        thread_id: The conversation thread ID used
    """
    response: str
    tools_used: list[str] = []
    source_docs: list[str] = []
    thread_id: str = ""


# ─── Endpoints ───

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns server status and list of connected tools.
    """
    tool_names = [t.name for t in app_state["tools"]] if app_state["tools"] else []
    return {
        "status": "healthy",
        "tools_connected": len(tool_names),
        "tool_names": tool_names,
    }


@app.get("/tools")
async def list_tools():
    """
    List all available MCP tools with their descriptions.
    """
    if not app_state["tools"]:
        return {"tools": []}

    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in app_state["tools"]
        ]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Flow:
      1. Receive user query + thread_id
      2. Invoke LangGraph chatbot (which may call MCP tools)
      3. Extract final response, tools used, and source docs
      4. Return structured response
    
    Args:
        request: ChatRequest with query and optional thread_id
        
    Returns:
        ChatResponse with response text, tools used, and source docs
    """
    chatbot = app_state["chatbot"]
    make_config = app_state["make_config"]

    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    config = make_config(request.thread_id)

    try:
        result = await chatbot.ainvoke(
            {"messages": [HumanMessage(content=request.query)]},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

    # ── Extract response details ──
    messages = result["messages"]
    
    # Final AI response (last message)
    final_response = messages[-1].content if messages else "No response generated."

    # Track which tools were used
    tools_used = []
    source_docs = []

    for msg in messages:
        # Check for tool calls in AI messages
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc["name"])

        # Check for RAG results in tool messages
        if isinstance(msg, ToolMessage):
            try:
                tool_result = json.loads(msg.content)
                if isinstance(tool_result, dict) and tool_result.get("found"):
                    source_docs.extend(tool_result.get("documents", []))
            except (json.JSONDecodeError, TypeError):
                pass

    return ChatResponse(
        response=final_response,
        tools_used=tools_used,
        source_docs=source_docs,
        thread_id=request.thread_id,
    )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint.
    Streams Server-Sent Events (SSE) indicating tool usage and AI tokens.
    """
    chatbot = app_state["chatbot"]
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    config = app_state["make_config"](request.thread_id)

    async def event_generator():
        try:
            # LangGraph astream_events streams node executions and LLM tokens
            async for event in chatbot.astream_events(
                {"messages": [HumanMessage(content=request.query)]},
                config=config,
                version="v2"
            ):
                kind = event["event"]
                
                if kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name']})}\n\n"
                    
                elif kind == "on_tool_end":
                    # Extract source docs from RAG tool
                    data = event.get("data", {})
                    output = data.get("output", {})
                    if hasattr(output, "content"):
                        try:
                            res = json.loads(output.content)
                            if isinstance(res, dict) and res.get("found"):
                                yield f"data: {json.dumps({'type': 'source_docs', 'docs': res.get('documents', [])})}\n\n"
                        except:
                            pass

                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        yield f"data: {json.dumps({'type': 'content', 'text': chunk.content})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    print(f"🌐 Starting API server on {host}:{port}")
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=True,
    )
