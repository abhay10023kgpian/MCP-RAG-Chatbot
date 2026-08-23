"""
rag_server.py — RAG Retrieval MCP Tool Server
===============================================
Exposes a FastMCP tool `retrieve_from_knowledge_base` that:
  1. Takes a user query string
  2. Searches the ChromaDB vector store for semantically similar documents
  3. Returns raw retrieved chunks (NOT synthesized — the client LLM does that)
  4. Returns {found: False} if no relevant documents exist

Based on: tempcodes/retrieve.py
Key changes: 
  - Removed internal LLM call — tool is pure retrieval.
  - Uses Google Gemini text-embedding-004 (lightweight cloud API, no PyTorch).

Run standalone:  python server/rag_server.py
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from fastmcp import FastMCP
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ─── Load .env from project root ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ─── Configuration ───
CHROMA_DIR = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_knowledge_base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ─── Initialize FastMCP Server ───
mcp = FastMCP("rag_knowledge_server")


@mcp.tool()
def retrieve_from_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for documents relevant to the user's query.
    
    This tool searches a ChromaDB vector store containing embedded documents
    (physics notes, theory text, etc.) and returns the most relevant chunks.
    
    Use this tool when the user asks about topics that might be covered in
    the stored knowledge base documents (e.g., physics concepts, motion,
    forces, acceleration, etc.).
    
    Args:
        query: The user's question or search query
        
    Returns:
        JSON string with structure:
        - found (bool): Whether relevant documents were found
        - query (str): The original query
        - documents (list): List of relevant text chunks (if found)
        - num_results (int): Number of documents returned
    """
    try:
        # ── Step 1: Initialize embedding model (Google Gemini, cloud API) ──
        embedding_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )

        # ── Step 2: Load existing ChromaDB ──
        if not Path(CHROMA_DIR).exists():
            return json.dumps({
                "found": False,
                "query": query,
                "documents": [],
                "num_results": 0,
                "error": "ChromaDB not initialized. Run embed_documents.py first."
            })

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=CHROMA_DIR
        )

        # ── Step 3: Similarity search with relevance scores ──
        results_with_scores = vector_store.similarity_search_with_relevance_scores(
            query, k=3
        )

        # ── Step 4: Filter by relevance threshold ──
        # Score > 0.3 means reasonably relevant (cosine similarity)
        RELEVANCE_THRESHOLD = 0.3
        relevant_docs = [
            {
                "content": doc.page_content,
                "score": round(score, 4),
                "metadata": doc.metadata
            }
            for doc, score in results_with_scores
            if score >= RELEVANCE_THRESHOLD
        ]

        if not relevant_docs:
            return json.dumps({
                "found": False,
                "query": query,
                "documents": [],
                "num_results": 0,
                "message": "No relevant documents found in the knowledge base for this query."
            })

        return json.dumps({
            "found": True,
            "query": query,
            "documents": [doc["content"] for doc in relevant_docs],
            "scores": [doc["score"] for doc in relevant_docs],
            "num_results": len(relevant_docs)
        })

    except Exception as e:
        return json.dumps({
            "found": False,
            "query": query,
            "documents": [],
            "num_results": 0,
            "error": f"Retrieval error: {str(e)}"
        })


if __name__ == "__main__":
    print(f"Starting RAG Knowledge Server...")
    print(f"   ChromaDB: {CHROMA_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Embedding: {EMBEDDING_MODEL}")
    mcp.run()
