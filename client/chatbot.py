"""
chatbot.py — LangGraph Stateful Chatbot with MCP Tool Calling
===============================================================
The main orchestrator. Uses LangGraph to build a stateful chatbot that:
  1. Receives user messages
  2. Decides whether to call an MCP tool or answer directly
  3. If tool call needed → executes tool → feeds result back to LLM
  4. Synthesizes final response using tool results + conversation context
  5. Maintains conversation memory via MemorySaver checkpointer

Architecture (LangGraph Flow):
  
  START → chat_node → [has tool calls?]
                          ├── YES → tool_node → chat_node (synthesize)
                          └── NO  → END

Based on: 
  - Langraph_tutorials/chatbot_backend.py (LangGraph structure)
  - tempcodes/client/mcp_client.py (tool calling logic)

Key improvement: Uses LangGraph conditional routing instead of manual
if/else tool dispatch. The LLM decides tool usage naturally.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# ─── Load .env from project root ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ─── State Schema ───
class ChatState(TypedDict):
    """
    State flowing through the LangGraph chatbot.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    category: str


# ─── System Prompt ───
# Kept for reference, but we use specialized prompts inside nodes now.
SYSTEM_PROMPT = "You are a helpful AI assistant with access to tools."


async def create_chatbot(tools: list = None):
    # ─── Initialize Groq LLM ───
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    llm = ChatGroq(
        api_key=groq_api_key,
        model=groq_model,
        temperature=0,
    )

    # ─── Agent Node ───
    system_prompt = SystemMessage(content=(
        "You are a helpful and intelligent assistant. "
        "You have access to tools for retrieving knowledge and doing math. "
        "If a user asks a factual question, ALWAYS use the retrieve_from_knowledge_base tool to search for the answer. "
        "If the tool returns no relevant documents, clearly state that you don't know based on the provided knowledge. "
        "Do NOT hallucinate facts outside the knowledge base. "
        "For simple greetings or small talk, respond naturally without calling tools."
    ))

    # Bind tools and add our stream_response tag so the backend streams this LLM's tokens
    llm_with_tools = llm.bind_tools(tools).with_config({"tags": ["stream_response"]})

    async def agent_node(state: ChatState) -> dict:
        messages = [system_prompt] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # ─── Routing ───
    def should_continue(state: ChatState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        # If the LLM decided to call a tool, route to the tool node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # ─── Build the Graph ───
    graph = StateGraph(ChatState)
    
    graph.add_node("agent", agent_node)
    if tools:
        graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    
    if tools:
        graph.add_conditional_edges("agent", should_continue)
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)

    # ─── Compile with Memory ───
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    def make_config(thread_id: str = "default") -> dict:
        return {"configurable": {"thread_id": thread_id}}

    return compiled, make_config


async def run_interactive():
    """
    Run the chatbot in interactive CLI mode.
    Useful for testing without the API/frontend.
    """
    from client.mcp_client import get_mcp_tools

    print("╔══════════════════════════════════════════╗")
    print("║   MCP RAG Chatbot — Interactive Mode     ║")
    print("╚══════════════════════════════════════════╝")
    print()

    # ─── Connect to MCP servers ───
    print("🔌 Connecting to MCP servers...")
    client, tools, named_tools = await get_mcp_tools()

    # ─── Create chatbot ───
    print("🤖 Creating chatbot...")
    chatbot, make_config = await create_chatbot(tools)
    config = make_config("interactive-session")

    print("\n✅ Ready! Type your questions (type 'quit' to exit)\n")
    print("─" * 50)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        print("\n🔄 Thinking...", end="", flush=True)

        try:
            result = await chatbot.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

            # Get the final AI response
            ai_message = result["messages"][-1]
            print(f"\r🤖 Assistant: {ai_message.content}")

            # Show if tools were used
            for msg in result["messages"]:
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"   🔧 Used tool: {tc['name']}")

        except Exception as e:
            print(f"\r❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(run_interactive())
