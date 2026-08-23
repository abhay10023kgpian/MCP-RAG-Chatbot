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

    # ─── Separate Tools ───
    math_tools = [t for t in tools if t.name.startswith("calculator")] if tools else []
    rag_tools = [t for t in tools if t.name == "retrieve_from_knowledge_base"] if tools else []
    
    math_llm = llm.bind_tools(math_tools) if math_tools else llm

    # ─── Define Graph Nodes ───
    async def classifier_node(state: ChatState) -> dict:
        last_msg = state["messages"][-1].content
        prompt = f"""Classify this user input into EXACTLY one of these categories:
- 'greeting': Casual conversation, hellos, thank yous
- 'math': Requests involving calculations or numbers
- 'factual': Requests asking for information, facts, general questions, or physics.

Input: "{last_msg}"
Category:"""
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        cat_text = response.content.lower()
        if "greeting" in cat_text: cat = "greeting"
        elif "math" in cat_text: cat = "math"
        else: cat = "factual"
        return {"category": cat}

    async def greeting_node(state: ChatState) -> dict:
        msg = AIMessage(content="Hello! I am your strict RAG assistant. How can I help you today?")
        return {"messages": [msg]}

    async def math_chat_node(state: ChatState) -> dict:
        response = await math_llm.ainvoke(state["messages"])
        return {"messages": [response]}

    async def force_rag_node(state: ChatState) -> dict:
        last_msg = state["messages"][-1].content
        tool_call = {
            "name": "retrieve_from_knowledge_base",
            "args": {"query": last_msg},
            "id": "forced_call_1"
        }
        msg = AIMessage(content="", tool_calls=[tool_call])
        return {"messages": [msg]}
        
    async def synthesize_node(state: ChatState) -> dict:
        strict_prompt = SystemMessage(content="Use ONLY the retrieved context. Do not invent facts. If the tool returned {'found': false}, you MUST reply 'I cannot answer this based on the provided documents.'")
        messages = [strict_prompt] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    async def evaluator_node(state: ChatState) -> dict:
        last_ai_message = state["messages"][-1].content
        tool_content = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage):
                tool_content = msg.content
                break
                
        critic_prompt = f"""You are a strict evaluator. 
Source Documents: {tool_content}
Generated Answer: {last_ai_message}

Does the Generated Answer contain any specific facts NOT present in the Source Documents? 
Reply strictly with "YES" or "NO"."""
        eval_response = await llm.ainvoke([HumanMessage(content=critic_prompt)])
        
        if "YES" in eval_response.content.upper():
            safe_response = AIMessage(content="I apologize, but my generated answer included outside knowledge, so I am withholding it to guarantee accuracy.")
            safe_response.id = state["messages"][-1].id 
            return {"messages": [safe_response]}
            
        return {"messages": []}

    # ─── Routing ───
    def route_after_classifier(state: ChatState) -> Literal["greeting_node", "math_chat_node", "force_rag_node"]:
        cat = state.get("category", "factual")
        if cat == "greeting": return "greeting_node"
        if cat == "math": return "math_chat_node"
        return "force_rag_node"
        
    def route_after_math_chat(state: ChatState) -> Literal["math_tool_node", "__end__"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "math_tool_node"
        return END

    # ─── Build the Graph ───
    graph = StateGraph(ChatState)
    
    graph.add_node("classifier_node", classifier_node)
    graph.add_node("greeting_node", greeting_node)
    graph.add_node("math_chat_node", math_chat_node)
    graph.add_node("force_rag_node", force_rag_node)
    graph.add_node("synthesize_node", synthesize_node)
    graph.add_node("evaluator_node", evaluator_node)
    
    if math_tools:
        graph.add_node("math_tool_node", ToolNode(math_tools))
    if rag_tools:
        graph.add_node("rag_tool_node", ToolNode(rag_tools))

    graph.add_edge(START, "classifier_node")
    graph.add_conditional_edges("classifier_node", route_after_classifier)
    graph.add_edge("greeting_node", END)
    
    if math_tools:
        graph.add_conditional_edges("math_chat_node", route_after_math_chat)
        graph.add_edge("math_tool_node", "math_chat_node")
    else:
        graph.add_edge("math_chat_node", END)
        
    if rag_tools:
        graph.add_edge("force_rag_node", "rag_tool_node")
        graph.add_edge("rag_tool_node", "synthesize_node")
    else:
        graph.add_edge("force_rag_node", "synthesize_node")
        
    graph.add_edge("synthesize_node", "evaluator_node")
    graph.add_edge("evaluator_node", END)

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
