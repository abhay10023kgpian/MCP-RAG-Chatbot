# 🏗️ Strict RAG LangGraph Prototype (Pre-Flight Classifier)

This document demonstrates how to implement a **Pre-flight Classifier** to completely prevent hallucinations and block outside knowledge in a LangGraph chatbot.

By adding a classifier *before* the main LLM call, we remove the LLM's freedom to bypass tools for factual queries.

## 1. Graph State Setup

```python
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    # We can store the category decided by the classifier
    category: str
```

## 2. The Nodes

### Node 1: Pre-flight Classifier
This node is the very first step. It asks a strict LLM to classify the query.
```python
async def classifier_node(state: ChatState):
    last_msg = state["messages"][-1].content
    prompt = f"""Classify this user input into EXACTLY one of these categories:
    - 'greeting': Casual conversation (hi, hello, thanks)
    - 'math': Requests involving calculation or numbers
    - 'factual': Requests asking for information, facts, or questions.

    Input: "{last_msg}"
    Category:"""
    
    # We use temperature=0 for deterministic output
    response = await llm.ainvoke(prompt)
    
    category = "factual"
    if "greeting" in response.content.lower(): category = "greeting"
    elif "math" in response.content.lower(): category = "math"
    
    return {"category": category}
```

### Node 2: Greeting & Math Paths
```python
async def greeting_node(state: ChatState):
    # Safe conversational fallback
    msg = AIMessage(content="Hello! I am your AI assistant. How can I help you today?")
    return {"messages": [msg]}

async def math_chat_node(state: ChatState):
    # LLM explicitly bound ONLY to math tools
    math_llm = llm.bind_tools(math_tools)
    response = await math_llm.ainvoke(state["messages"])
    return {"messages": [response]}
```

### Node 3: Forced RAG Node (Factual Path)
For factual queries, we don't let the LLM decide. We manually construct a tool call!
```python
async def force_rag_node(state: ChatState):
    last_msg = state["messages"][-1].content
    
    # We forge an AI message that contains a tool call to our RAG tool
    tool_call = {
        "name": "retrieve_from_knowledge_base",
        "args": {"query": last_msg},
        "id": "forced_call_123"
    }
    msg = AIMessage(content="", tool_calls=[tool_call])
    return {"messages": [msg]}
```

### Node 4: The Tool Node
Executes the tool calls (either math or the forced RAG call).
```python
from langgraph.prebuilt import ToolNode
tool_node = ToolNode(all_tools)
```

### Node 5: The Synthesizer & Evaluator
```python
async def synthesize_node(state: ChatState):
    strict_prompt = SystemMessage(content="Use ONLY the retrieved context. Do not invent facts.")
    messages = [strict_prompt] + state["messages"]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}

async def evaluator_node(state: ChatState):
    # Similar to the previous prototype:
    # 1. Check if the synthesize_node hallucinated facts not in ToolMessage.
    # 2. If YES, overwrite with "I cannot answer this based on the knowledge base."
    ...
```

## 3. Conditional Routing Logic

The routing is entirely controlled by the classifier category.

```python
def route_after_classifier(state: ChatState) -> Literal["greeting_node", "math_chat_node", "force_rag_node"]:
    cat = state.get("category", "factual")
    if cat == "greeting": return "greeting_node"
    if cat == "math": return "math_chat_node"
    return "force_rag_node"

def route_after_math_chat(state: ChatState) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return "__end__"
```

## 4. Building the Graph

```python
graph = StateGraph(ChatState)

graph.add_node("classifier_node", classifier_node)
graph.add_node("greeting_node", greeting_node)
graph.add_node("math_chat_node", math_chat_node)
graph.add_node("force_rag_node", force_rag_node)
graph.add_node("tool_node", tool_node)
graph.add_node("synthesize_node", synthesize_node)
graph.add_node("evaluator_node", evaluator_node)

# Step 1: Always Classify
graph.add_edge(START, "classifier_node")
graph.add_conditional_edges("classifier_node", route_after_classifier)

# Step 2a: Greeting ends immediately
graph.add_edge("greeting_node", END)

# Step 2b: Math path (allows tools or direct answer)
graph.add_conditional_edges("math_chat_node", route_after_math_chat)

# Step 2c: Factual path (forces RAG tool)
graph.add_edge("force_rag_node", "tool_node")

# Step 3: Tool Node routes back to Synthesizer
# (For simplicity in this prototype, we assume all tools route to synthesize. 
# In reality, math tools might route back to math_chat_node).
graph.add_edge("tool_node", "synthesize_node")

# Step 4: Always evaluate factual synthesis
graph.add_edge("synthesize_node", "evaluator_node")
graph.add_edge("evaluator_node", END)

compiled = graph.compile()
```
