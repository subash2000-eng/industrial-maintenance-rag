"""
Phase 1 - Embedder
Cloud version using HuggingFace API for embeddings.
Zero RAM — no local model needed.
"""

import json
import os
import time
import requests as req
from pathlib import Path
from tqdm    import tqdm
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH",  "./data/chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "maintenance_manuals")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
HF_API_TOKEN    = os.getenv("HF_API_TOKEN", "")
BATCH_SIZE      = 16

HF_API_URL = (
    f"https://api-inference.huggingface.co/pipeline/"
    f"feature-extraction/sentence-transformers/{EMBEDDING_MODEL}"
)


def get_embeddings_via_api(texts: list) -> list:
    """Get embeddings from HuggingFace free API."""
    headers  = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload  = {
        "inputs" : texts,
        "options": {"wait_for_model": True}
    }

    for attempt in range(3):
        try:
            response = req.post(
                HF_API_URL,
                headers=headers,
                json   =payload,
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                # Normalize output shape
                embeddings = []
                for emb in result:
                    if isinstance(emb[0], list):
                        embeddings.append(emb[0])
                    else:
                        embeddings.append(emb)
                return embeddings
            elif response.status_code == 503:
                print(f"  ⏳ Model loading, waiting 20s...")
                time.sleep(20)
            else:
                print(f"  ⚠️  API error: {response.text}")
                time.sleep(5)
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt+1} failed: {e}")
            time.sleep(5)

    raise Exception("HuggingFace API failed after 3 attempts")


def load_embedding_model(model_name: str = EMBEDDING_MODEL):
    """
    Returns None — we use HF API instead of local model.
    Kept for API compatibility.
    """
    print(f"\n🔄 Embedding mode: HuggingFace API")
    print(f"  Model: {model_name}")
    print(f"  ✅ No local model — zero RAM usage")
    return None


def get_chroma_client(db_path: str = CHROMA_DB_PATH):
    """Connect to ChromaDB."""
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path    =str(db_path),
        settings=Settings(anonymized_telemetry=False)
    )


def get_or_create_collection(
    client,
    collection_name: str = COLLECTION_NAME
):
    """Get or create ChromaDB collection."""
    collection = client.get_or_create_collection(
        name    =collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"  📁 Collection: '{collection_name}'")
    print(f"  📊 Documents : {collection.count()}")
    return collection


def embed_and_store_chunks(
    chunks    : list,
    collection,
    model             = None,
    batch_size: int   = BATCH_SIZE
) -> None:
    """
    Generate embeddings via HF API and store in ChromaDB.
    """
    existing     = collection.get(include=[])
    existing_ids = set(existing['ids'])
    new_chunks   = [
        c for c in chunks
        if c['chunk_id'] not in existing_ids
    ]

    if not new_chunks:
        print("  ✅ All chunks already embedded")
        return

    print(f"\n🔢 Embedding {len(new_chunks)} chunks via HF API...")

    for i in tqdm(
        range(0, len(new_chunks), batch_size),
        desc="  Embedding"
    ):
        batch  = new_chunks[i : i + batch_size]
        texts  = [chunk['text'] for chunk in batch]

        # Get embeddings from HF API
        embeddings = get_embeddings_via_api(texts)

        ids   = [chunk['chunk_id'] for chunk in batch]
        metas = [
            {
                "manual_name"  : chunk['manual_name'],
                "page_number"  : chunk['page_number'],
                "total_pages"  : chunk['total_pages'],
                "section_title": chunk['section_title'],
                "chunk_index"  : chunk['chunk_index'],
                "token_count"  : chunk['token_count'],
                "char_count"   : chunk['char_count']
            }
            for chunk in batch
        ]

        collection.add(
            ids       =ids,
            documents =texts,
            embeddings=embeddings,
            metadatas =metas
        )

        # Small delay to respect API rate limits
        time.sleep(0.5)

    print(f"\n✅ ChromaDB: {collection.count()} total chunks")


def verify_embedding(collection, model=None,
                     test_query: str = "motor overheating"):
    """Quick test of embedding + search."""
    from vector_search import vector_search
    print(f"\n🧪 Test query: '{test_query}'")
    results = vector_search(test_query, collection, None, top_k=3)
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r['manual_name']} p.{r['page_number']}")
        print(f"       {r['text'][:100]}...")


def run_embedding_pipeline(
    chunks_path    : str,
    db_path        : str = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_NAME,
    model_name     : str = EMBEDDING_MODEL
) -> tuple:
    """Full embedding pipeline."""
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"  📦 Loaded {len(chunks)} chunks")

    model      = load_embedding_model(model_name)
    client     = get_chroma_client(db_path)
    collection = get_or_create_collection(
        client, collection_name
    )
    embed_and_store_chunks(chunks, collection, model)
    return collection, model