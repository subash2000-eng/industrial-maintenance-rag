"""
Phase 2 - Vector Search
Cloud version using HuggingFace API for embeddings.
Zero RAM usage for embedding model.
"""

import os
import sys
import requests as req
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "retrieval"))
sys.path.insert(0, str(BASE_DIR / "ingestion"))

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "maintenance_manuals")
HF_API_TOKEN    = os.getenv("HF_API_TOKEN", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K           = int(os.getenv("TOP_K_RETRIEVAL", 20))

# HuggingFace API endpoint for embeddings
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/{EMBEDDING_MODEL}"


def get_embeddings_via_api(texts: list) -> list:
    """
    Get embeddings using HuggingFace Inference API.
    Free — no local model needed — zero RAM.
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs"    : texts,
        "options"   : {"wait_for_model": True}
    }

    response = req.post(
        HF_API_URL,
        headers=headers,
        json   =payload,
        timeout=30
    )

    if response.status_code != 200:
        raise Exception(
            f"HuggingFace API error: {response.text}"
        )

    return response.json()


def get_chroma_client():
    """Connect to ChromaDB."""
    db_path = Path(CHROMA_DB_PATH)
    db_path.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path    =str(db_path),
        settings=Settings(anonymized_telemetry=False)
    )


def load_vector_components():
    """
    Load vector search components.
    Returns (None, collection) — model is None
    because we use HF API instead.
    """
    print("\n🔄 Loading vector search components...")

    client     = get_chroma_client()
    collection = client.get_or_create_collection(
        name    =COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"  ✅ ChromaDB connected: {collection.count()} chunks")
    print(f"  ✅ Using HuggingFace API for embeddings")

    return None, collection


def vector_search(
    query         : str,
    collection    ,
    model         ,
    top_k         : int = TOP_K,
    manual_filter : str = None
) -> list:
    """
    Semantic search using HuggingFace API embeddings.
    """
    # Get query embedding via API
    try:
        embedding_result = get_embeddings_via_api([query])
        # Result shape: [[...embedding...]]
        query_embedding  = embedding_result[0]

        # If nested list — flatten one level
        if isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]

    except Exception as e:
        print(f"  ⚠️  HF API error: {e}")
        print("  Falling back to keyword search only")
        return []

    # Optional manual filter
    where_filter = None
    if manual_filter and manual_filter != "All Manuals":
        where_filter = {"manual_name": {"$eq": manual_filter}}

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results        =min(top_k, max(1, collection.count())),
        include          =['documents', 'metadatas', 'distances'],
        where            =where_filter
    )

    # Format results
    formatted = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        similarity = round(1 - dist, 4)
        formatted.append({
            "text"         : doc,
            "manual_name"  : meta['manual_name'],
            "page_number"  : meta['page_number'],
            "section_title": meta['section_title'],
            "token_count"  : meta['token_count'],
            "score"        : similarity,
            "source"       : "vector"
        })

    return formatted