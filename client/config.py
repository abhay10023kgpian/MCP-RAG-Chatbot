"""
config.py — Centralized Configuration Loader
=============================================
Single source of truth for all environment variables and constants.
Loads from .env file and provides validated access to all config values.

Used by: client/mcp_client.py, client/chatbot.py, api/server.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ─── Locate and load .env from project root ───
# Walk up from this file's directory to find the project root .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    print(f"⚠️  Warning: .env file not found at {ENV_PATH}", file=sys.stderr)
    print("   Copy .env.example to .env and fill in your GROQ_API_KEY", file=sys.stderr)


# ─── Groq LLM Configuration ───
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─── Embedding Model (Google Gemini — lightweight cloud API, no PyTorch) ───
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

# ─── ChromaDB ───
CHROMA_DIR = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_knowledge_base")

# ─── Storage (raw documents for embedding) ───
STORAGE_DIR = str(PROJECT_ROOT / "storage")

# ─── API Server ───
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ─── MCP Server Paths (resolved relative to project root) ───
PYTHON_EXECUTABLE = sys.executable  # Current Python interpreter
RAG_SERVER_PATH = str(PROJECT_ROOT / "server" / "rag_server.py")
MATH_SERVER_PATH = str(PROJECT_ROOT / "server" / "math_server.py")


def validate_config():
    """
    Validates that all critical configuration values are set.
    Call this at startup to fail fast if config is missing.
    
    Returns:
        bool: True if all required config is present
        
    Raises:
        SystemExit: If critical config (GROQ_API_KEY) is missing
    """
    errors = []

    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        errors.append("GROQ_API_KEY is not set. Add it to your .env file.")

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
        errors.append("GOOGLE_API_KEY is not set. Get one at https://aistudio.google.com/apikey")

    if errors:
        print("\n❌ Configuration Errors:", file=sys.stderr)
        for err in errors:
            print(f"   • {err}", file=sys.stderr)
        print(f"\n   Edit: {ENV_PATH}\n", file=sys.stderr)
        sys.exit(1)

    return True


def print_config():
    """
    Prints current configuration for debugging.
    Masks sensitive values (API keys).
    """
    masked_key = GROQ_API_KEY[:8] + "..." if len(GROQ_API_KEY) > 8 else "NOT SET"
    masked_google = GOOGLE_API_KEY[:8] + "..." if len(GOOGLE_API_KEY) > 8 else "NOT SET"
    print("┌─────────────────────────────────────────┐")
    print("│       MCP RAG Chatbot — Config          │")
    print("├─────────────────────────────────────────┤")
    print(f"│ GROQ_API_KEY:    {masked_key:<22} │")
    print(f"│ GROQ_MODEL:      {GROQ_MODEL:<22} │")
    print(f"│ GOOGLE_API_KEY:  {masked_google:<22} │")
    print(f"│ EMBEDDING_MODEL: {EMBEDDING_MODEL.split('/')[-1]:<22} │")
    print(f"│ CHROMA_DIR:      {Path(CHROMA_DIR).name:<22} │")
    print(f"│ COLLECTION:      {COLLECTION_NAME:<22} │")
    print("└─────────────────────────────────────────┘")
