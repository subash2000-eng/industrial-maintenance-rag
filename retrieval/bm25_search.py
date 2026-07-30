"""
Phase 2 - BM25 Keyword Search
Exact keyword search using BM25 algorithm.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))

import json
import os
import re
import pickle
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
CHUNKS_FILE     = BASE_DIR / "data/processed_chunks/all_chunks.json"
BM25_INDEX_PATH = BASE_DIR / "data/processed_chunks/bm25_index.pkl"
TOP_K           = int(os.getenv("TOP_K_RETRIEVAL", 20))


def tokenize(text: str) -> list:
    """
    Convert text to list of tokens for BM25.
    Preserves fault codes like E-204 and ERR_001.
    """
    text   = text.lower()
    tokens = re.findall(r'[a-z0-9_-]+', text)
    return tokens


def build_bm25_index(chunks: list) -> tuple:
    """
    Build BM25 index from all chunks.
    """
    print(f"\n🔨 Building BM25 index from {len(chunks)} chunks...")

    tokenized_corpus = [
        tokenize(chunk['text'])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  ✅ BM25 index built successfully")
    return bm25, chunks


def save_bm25_index(bm25, chunks: list) -> None:
    """
    Save BM25 index and chunks to disk.
    """
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"  💾 BM25 index saved: {BM25_INDEX_PATH.name}")


def load_bm25_index() -> tuple:
    """
    Load BM25 index from disk.
    """
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_INDEX_PATH}\n"
            "Run this file directly first to build the index."
        )

    with open(BM25_INDEX_PATH, 'rb') as f:
        data = pickle.load(f)

    print(f"  ✅ BM25 index loaded: {len(data['chunks'])} chunks")
    return data['bm25'], data['chunks']

def bm25_search(
    query  : str,
    bm25   : BM25Okapi,
    chunks : list,
    top_k  : int = TOP_K
) -> list:
    """
    Search chunks using BM25 keyword matching.
    Works correctly with filtered chunk subsets.
    """
    if not chunks:
        return []

    tokenized_query = tokenize(query)

    # Build a LOCAL BM25 index from the given chunks
    # This fixes the index mismatch when chunks are filtered
    local_corpus = [tokenize(c['text']) for c in chunks]
    local_bm25   = BM25Okapi(local_corpus)
    scores       = local_bm25.get_scores(tokenized_query)

    # Get indices of top-k highest scores
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    # Build results list
    results = []
    for idx in top_indices:
        score = scores[idx]

        # Skip chunks with zero score
        if score <= 0:
            continue

        # Safe index check
        if idx >= len(chunks):
            continue

        chunk = chunks[idx]
        results.append({
            "text"         : chunk['text'],
            "manual_name"  : chunk['manual_name'],
            "page_number"  : chunk['page_number'],
            "section_title": chunk['section_title'],
            "token_count"  : chunk['token_count'],
            "score"        : round(float(score), 4),
            "source"       : "bm25"
        })

    return results


# ── Run this file directly to test ──────────────────────
if __name__ == "__main__":

    if not CHUNKS_FILE.exists():
        print("❌ all_chunks.json not found")
        print("   Run ingest_pipeline.py first")
    else:
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"  📦 Loaded {len(chunks)} chunks")

        bm25, chunks = build_bm25_index(chunks)
        save_bm25_index(bm25, chunks)

        test_queries = [
            "pressure relief valve fault",
            "motor bearing temperature warning",
            "compressor oil level maintenance"
        ]

        for query in test_queries:
            results = bm25_search(query, bm25, chunks, top_k=3)

            print(f"\n{'='*58}")
            print(f"  Query  : '{query}'")
            print(f"  Results: {len(results)}")
            print(f"{'='*58}\n")

            for i, r in enumerate(results):
                print(f"  [{i+1}] Score  : {r['score']}")
                print(f"       Manual : {r['manual_name']}")
                print(f"       Page   : {r['page_number']}")
                print(f"       Text   : {r['text'][:200]}...")
                print()