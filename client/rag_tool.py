"""
rag_tool.py — Direct RAG Retrieval Tool (No MCP Subprocess)
=============================================================
Replaces the MCP-based rag_server.py subprocess with a direct
LangChain @tool that calls ChromaDB in-process.

This eliminates the stdio subprocess failure on Render/cloud deployments.

The vector store is initialized once (lazy singleton) and reused across
all requests, avoiding repeated embedding model instantiation.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
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

# ─── Lazy Singleton Vector Store ───
_vector_store = None


def _get_vector_store():
    """
    Initialize and cache the ChromaDB vector store.
    Called once on first retrieval request, reused thereafter.
    """
    global _vector_store
    if _vector_store is None:
        print(f"[rag_tool] Initializing vector store...")
        print(f"  ChromaDB dir: {CHROMA_DIR}")
        print(f"  Collection:   {COLLECTION_NAME}")
        print(f"  Embedding:    {EMBEDDING_MODEL}")

        if not Path(CHROMA_DIR).exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found: {CHROMA_DIR}. "
                "Run embed_documents.py first."
            )

        embedding_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=CHROMA_DIR,
        )
        # Quick sanity check
        count = _vector_store._collection.count()
        print(f"  Vectors loaded: {count}")
    return _vector_store


@tool
def retrieve_from_knowledge_base(query: str) -> str:
    """Search the knowledge base for documents relevant to the user's query.

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
        vs = _get_vector_store()

        # ── Similarity search with relevance scores ──
        results_with_scores = vs.similarity_search_with_relevance_scores(
            query, k=3
        )

        # ── Filter by relevance threshold ──
        RELEVANCE_THRESHOLD = 0.3
        relevant_docs = [
            {
                "content": doc.page_content,
                "score": round(score, 4),
                "metadata": doc.metadata,
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
                "message": "No relevant documents found in the knowledge base for this query.",
            })

        return json.dumps({
            "found": True,
            "query": query,
            "documents": [doc["content"] for doc in relevant_docs],
            "scores": [doc["score"] for doc in relevant_docs],
            "num_results": len(relevant_docs),
        })

    except Exception as e:
        return json.dumps({
            "found": False,
            "query": query,
            "documents": [],
            "num_results": 0,
            "error": f"Retrieval error: {str(e)}",
        })
