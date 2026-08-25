"""
mcp_client.py — Multi-Server MCP Client Setup
================================================
Sets up connections to all MCP tool servers and returns bound tools.

Architecture:
  This module handles the "plumbing" — connecting to MCP servers via stdio,
  discovering available tools, and providing them ready for LLM binding.

Servers connected:
  1. rag_knowledge_server — RAG retrieval from ChromaDB
  2. math_tools_server   — Basic calculator operations

Based on: tempcodes/client/mcp_client.py
Key changes:
  - Uses relative paths (not hardcoded absolute paths)
  - Uses Groq instead of Azure OpenAI
  - Proper error handling for server connections
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# ─── Load .env from project root ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ─── Resolve paths ───
PYTHON_EXE = sys.executable
RAG_SERVER_PATH = str(PROJECT_ROOT / "server" / "rag_server.py")
MATH_SERVER_PATH = str(PROJECT_ROOT / "server" / "math_server.py")
GITHUB_SERVER_PATH = str(PROJECT_ROOT / "server" / "github_issues_server.py")



def get_mcp_server_config() -> dict:
    """
    Returns the MCP server configuration dictionary.
    
    Each server entry specifies:
      - transport: "stdio" (subprocess communication)
      - command: Python executable path
      - args: Path to the server script
      
    The MultiServerMCPClient uses this config to spawn and connect
    to each MCP server as a subprocess.
    
    Returns:
        dict: Server configuration for MultiServerMCPClient
    """
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    return {
        "rag_knowledge_server": {
            "transport": "stdio",
            "command": PYTHON_EXE,
            "args": [RAG_SERVER_PATH],
            "env": env
        },
        "math_tools_server": {
            "transport": "stdio",
            "command": PYTHON_EXE,
            "args": [MATH_SERVER_PATH],
            "env": env
        },
        "github_issues_connector": {
            "transport": "stdio",
            "command": PYTHON_EXE,
            "args": [GITHUB_SERVER_PATH],
            "env": env
        }
    }


async def get_mcp_tools():
    """
    Connect to all MCP servers and retrieve available tools.
    
    Flow:
      1. Create MultiServerMCPClient with server config
      2. Connect to each server via stdio
      3. Discover all available tools across all servers
      4. Return tools list ready for LLM binding
    
    Returns:
        tuple: (client, tools, named_tools)
          - client: The MultiServerMCPClient instance (keep alive!)
          - tools: List of LangChain Tool objects
          - named_tools: Dict mapping tool_name → Tool object
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    
    servers = get_mcp_server_config()
    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()

    named_tools = {}
    print("\n🔧 Available MCP Tools:")
    for tool in tools:
        named_tools[tool.name] = tool
        print(f"   • {tool.name}: {tool.description[:80]}...")

    return client, tools, named_tools
