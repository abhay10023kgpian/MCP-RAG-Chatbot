"""
app.py — Streamlit Chat Frontend
==================================
Chat interface for the MCP RAG Chatbot.

Features:
  - Conversation history with session state
  - Shows when tools are being called (spinner)
  - Displays source documents when RAG retrieval is used
  - Displays which tool was used for each response
  - Thread-based conversation (persists memory via API)

Based on: Langraph_tutorials/Chatbot_frontend.py
Key changes: Calls FastAPI backend instead of direct chatbot invocation,
             shows tool usage and source documents.

Usage:
  streamlit run frontend/app.py
"""

import os
import uuid
import json
import requests
import streamlit as st

# ─── Configuration ───
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
THREAD_ID = "streamlit-session"

# ─── Page Config ───
st.set_page_config(
    page_title="MCP RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ─── Custom Styling ───
st.markdown("""
<style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
    .tool-badge {
        background-color: #1a1a2e;
        color: #e94560;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
    }
    .source-doc {
        background-color: #16213e;
        border-left: 3px solid #0f3460;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
        font-size: 0.85em;
    }
    .header-container {
        text-align: center;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.markdown("""
<div class="header-container">
    <h1>🤖 MCP RAG Chatbot</h1>
    <p style="color: #888;">Multi-Server • Tool-Calling • Knowledge Retrieval</p>
</div>
""", unsafe_allow_html=True)

# ─── Check API Health ───
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    tool_names = health.get("tool_names", [])
    st.sidebar.success(f"✅ API Connected — {health.get('tools_connected', 0)} tools")
    st.sidebar.markdown("**Available Tools:**")
    for name in tool_names:
        st.sidebar.markdown(f"- `{name}`")
except requests.exceptions.ConnectionError:
    st.sidebar.error("❌ API not running. Start with:\n```\nuvicorn api.server:app --port 8000\n```")
    tool_names = []

# ─── Sidebar Info ───
st.sidebar.markdown("---")
st.sidebar.markdown("### How it works")
st.sidebar.markdown("""
1. Your message goes to the **LLM** (Groq)
2. LLM decides if a **tool** is needed
3. If yes → calls **MCP tool server**
4. Tool retrieves from **ChromaDB**
5. LLM synthesizes the **final answer**
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Try asking:")
st.sidebar.markdown("""
- *"What is Newton's second law?"*
- *"Explain acceleration"*
- *"What is 25 + 37?"*
- *"Tell me about quantum physics"*  
  *(not in knowledge base)*
""")

# ─── Initialize Session State ───
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# ─── Display Chat History ───
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show tool badges
        if message.get("tools_used"):
            tools_text = " ".join([f"`🔧 {t}`" for t in message["tools_used"]])
            st.caption(f"Tools used: {tools_text}")

        # Show source documents
        if message.get("source_docs"):
            with st.expander("📄 Source Documents", expanded=False):
                for i, doc in enumerate(message["source_docs"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(doc[:500] + ("..." if len(doc) > 500 else ""))
                    st.markdown("---")

# ─── Chat Input ───
user_input = st.chat_input("Ask me anything...")

if user_input:
    # ── Add user message ──
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input,
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Call API ──
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        
        full_response = ""
        tools_used = []
        source_docs = []
        
        # Initial status
        status_placeholder.markdown("🔄 Thinking...")

        try:
            response = requests.post(
                f"{API_URL}/chat/stream",
                json={
                    "query": user_input,
                    "thread_id": THREAD_ID,
                },
                stream=True,
                timeout=60,
            )
            
            if response.status_code != 200:
                raise Exception(f"API Error: {response.text}")
                
            status_cleared = False
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        try:
                            data = json.loads(data_str)
                            
                            if data["type"] == "tool_start":
                                tool_name = data["tool"]
                                tools_used.append(tool_name)
                                status_placeholder.markdown(f"🛠️ Using tool: `{tool_name}`...")
                                
                            elif data["type"] == "source_docs":
                                source_docs.extend(data["docs"])
                                
                            elif data["type"] == "content":
                                if not status_cleared:
                                    status_placeholder.empty()
                                    status_cleared = True
                                full_response += data["text"]
                                message_placeholder.markdown(full_response + "▌")
                                
                            elif data["type"] == "done":
                                break
                                
                            elif data["type"] == "error":
                                st.error(data["message"])
                                break
                        except Exception as e:
                            print(f"SSE Parse Error: {e} | Data: {data_str}")

            # Finalize response UI
            message_placeholder.markdown(full_response)
            status_placeholder.empty()

            # Show tool badges
            if tools_used:
                tools_text = " ".join([f"`🔧 {t}`" for t in tools_used])
                st.caption(f"Tools used: {tools_text}")

            # Show source documents
            if source_docs:
                with st.expander("📄 Source Documents", expanded=False):
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc[:500] + ("..." if len(doc) > 500 else ""))
                        st.markdown("---")

            # Save to history
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": full_response,
                "tools_used": tools_used,
                "source_docs": source_docs,
            })

        except requests.exceptions.ConnectionError:
            error_msg = "❌ Cannot connect to API. Make sure the server is running."
            st.error(error_msg)
            status_placeholder.empty()
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": error_msg,
            })
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            status_placeholder.empty()
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": error_msg,
            })
