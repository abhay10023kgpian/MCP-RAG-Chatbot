"""
embed_documents.py — Document Embedding Pipeline
==================================================
One-time script (or run on startup) to embed text documents into ChromaDB.

Pipeline Flow:
  1. Reads all .txt files from storage/ directory
  2. Splits text into chunks (1000 chars, 50 overlap)
  3. Embeds using Google Gemini text-embedding-004 (cloud API, lightweight)
  4. Stores vectors in ChromaDB at ./chroma_db

Based on: tempcodes/rag_pipeline.py
Key change: Uses Google Gemini embeddings (no PyTorch, ~10MB vs ~2GB).

Usage:
  python embed_documents.py                    # Embed all .txt files in storage/
  python embed_documents.py --file myfile.txt  # Embed a specific file
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ─── Load .env ───
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ─── Configuration ───
CHROMA_DIR = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_knowledge_base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
STORAGE_DIR = str(PROJECT_ROOT / "storage")


def load_text_file(filepath: str) -> str:
    """
    Load a text file and return its contents.
    
    Args:
        filepath: Path to the .txt file
        
    Returns:
        File contents as a string
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks for embedding.
    
    Uses RecursiveCharacterTextSplitter which tries to split on
    natural boundaries (paragraphs -> sentences -> words).
    
    Args:
        text: Raw text content
        
    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", ", ", " "]
    )
    return splitter.split_text(text)


def main():
    parser = argparse.ArgumentParser(description="Embed documents into ChromaDB")
    parser.add_argument(
        "--file", 
        type=str, 
        default=None,
        help="Specific file to embed (relative to storage/). Default: all .txt files."
    )
    args = parser.parse_args()

    print("=" * 44)
    print("   Document Embedding Pipeline")
    print("=" * 44)
    print(f"  Storage dir:  {STORAGE_DIR}")
    print(f"  ChromaDB dir: {CHROMA_DIR}")
    print(f"  Collection:   {COLLECTION_NAME}")
    print(f"  Embedding:    {EMBEDDING_MODEL}")
    print()

    # Validate API key
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
        print("ERROR: GOOGLE_API_KEY is not set in .env")
        print("  Get one at: https://aistudio.google.com/apikey")
        sys.exit(1)

    # Determine which files to embed
    if args.file:
        files = [Path(STORAGE_DIR) / args.file]
        if not files[0].exists():
            print(f"ERROR: File not found: {files[0]}")
            sys.exit(1)
    else:
        storage_path = Path(STORAGE_DIR)
        if not storage_path.exists():
            print(f"ERROR: Storage directory not found: {STORAGE_DIR}")
            print("   Create the directory and add .txt files to embed.")
            sys.exit(1)
        files = list(storage_path.glob("*.txt"))
        if not files:
            print(f"ERROR: No .txt files found in {STORAGE_DIR}")
            sys.exit(1)

    print(f"  Found {len(files)} file(s) to embed:\n")

    all_chunks = []
    all_sources = []

    for filepath in files:
        print(f"  -- Processing: {filepath.name}")
        text = load_text_file(str(filepath))
        print(f"     Characters: {len(text):,}")
        
        chunks = split_text(text)
        print(f"     Chunks:     {len(chunks)}")
        
        all_chunks.extend(chunks)
        all_sources.extend([filepath.name] * len(chunks))

    print(f"\n  Total chunks to embed: {len(all_chunks)}")
    
    # Add metadata
    metadatas = [
        {"source": source, "chunk_index": i}
        for i, source in enumerate(all_sources)
    ]

    print(f"  Initializing Google Gemini embedding model...")
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    print(f"  Embedding and storing in ChromaDB in batches (Free Tier Rate Limit Protection)...")
    
    # Process in batches to avoid 429 RESOURCE_EXHAUSTED (Free tier limit: 100 req/min)
    import time
    batch_size = 80  # Under the 100 req/min limit
    vector_store = None
    
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        print(f"     Embedding batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1} ({len(batch_chunks)} chunks)...")
        
        if vector_store is None:
            vector_store = Chroma.from_texts(
                texts=batch_chunks,
                collection_name=COLLECTION_NAME,
                embedding=embedding_model,
                persist_directory=CHROMA_DIR,
                metadatas=batch_metadatas
            )
        else:
            vector_store.add_texts(texts=batch_chunks, metadatas=batch_metadatas)
            
        if i + batch_size < len(all_chunks):
            print("     Sleeping 65 seconds for Google API 100 req/min rate limit...")
            time.sleep(65)

    # Verify
    collection = vector_store._collection
    results = collection.get(include=["embeddings", "documents"])
    print(f"\n  Embedding complete!")
    print(f"     Vectors stored:          {len(results['documents'])}")
    print(f"     Embedding dimensions:    {len(results['embeddings'][0])}")
    print(f"     First document preview:  {results['documents'][0][:100]}...")
    print(f"\n  ChromaDB saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
