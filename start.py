"""
start.py — Production Startup Script (for Render / cloud deployment)
=====================================================================
This script is the entry point for deployment on platforms like Render.

Startup sequence:
  1. Embeds documents from storage/ into ChromaDB (handles ephemeral disk)
  2. Starts the FastAPI server with uvicorn

Usage in Render:
  Build Command:  pip install -r requirements.txt
  Start Command:  python start.py

Why re-embed on startup?
  Render free tier has ephemeral disk — chroma_db/ is lost on restart.
  This script regenerates it from storage/ files on every cold start.
  Takes ~30s for typical document sets.
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_embedding():
    """
    Run the embedding pipeline ONLY if ChromaDB doesn't exist.
    Pre-embedded chroma_db/ is committed to git and shipped with the code.
    Only re-embeds as a fallback if chroma_db/ is completely missing.
    """
    chroma_dir = Path(os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db")))
    
    if chroma_dir.exists() and (chroma_dir / "chroma.sqlite3").exists():
        print("[start.py] Pre-embedded ChromaDB found — using existing vectors.")
        return

    print("[start.py] ChromaDB not found — running embedding pipeline as fallback...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "embed_documents.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print("[start.py] WARNING: Embedding failed. Chatbot will run without RAG.", file=sys.stderr)
    else:
        print("[start.py] Embedding complete.")


def start_server():
    """
    Start the FastAPI server with uvicorn.
    """
    host = os.getenv("API_HOST", "0.0.0.0")
    port = os.getenv("PORT", os.getenv("API_PORT", "8000"))  # Render sets PORT env var

    print(f"[start.py] Starting API server on {host}:{port}")
    
    os.execvp(
        sys.executable,
        [
            sys.executable, "-m", "uvicorn",
            "api.server:app",
            "--host", host,
            "--port", str(port),
        ]
    )


if __name__ == "__main__":
    run_embedding()
    start_server()

