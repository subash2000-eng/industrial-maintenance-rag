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

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_FILE     = BASE_DIR / "data/processed_chunks/all_chunks.json"
BM25_INDEX_PATH = BASE_DIR / "data/processed_chunks/bm25_index.pkl"
TOP_K           = int(os.getenv("TOP_K_RETRIEVAL", 20))


def tokenize(text: str) -> list:
    return re.findall(r'[a-z0-9_-]+', text.lower())


def build_bm25_index(chunks: list) -> tuple:
    print(f"Building BM25 index from {len(chunks)} chunks...")
    tokenized_corpus = [tokenize(chunk['text']) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 index built successfully")
    return bm25, chunks


def save_bm25_index(bm25, chunks: list) -> None:
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"BM25 index saved: {BM25_INDEX_PATH.name}")


def load_bm25_index() -> tuple:
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found. Run bm25_search.py first."
        )
    with open(BM25_INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
    print(f"BM25 index loaded: {len(data['chunks'])} chunks")
    return data['bm25'], data['chunks']


def bm25_search(
    query : str,
    bm25  : BM25Okapi,
    chunks: list,
    top_k : int = TOP_K
) -> list:
    if not chunks:
        return []

    tokenized_query = tokenize(query)
    local_bm25      = BM25Okapi([tokenize(c['text']) for c in chunks])
    scores          = local_bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append({
            "text"         : chunk['text'],
            "manual_name"  : chunk['manual_name'],
            "page_number"  : chunk['page_number'],
            "section_title": chunk['section_title'],
            "token_count"  : chunk['token_count'],
            "score"        : round(float(scores[idx]), 4),
            "source"       : "bm25"
        })

    return results


if __name__ == "__main__":
    if not CHUNKS_FILE.exists():
        print("Run ingest_pipeline.py first.")
    else:
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        bm25, chunks = build_bm25_index(chunks)
        save_bm25_index(bm25, chunks)

        results = bm25_search("motor bearing temperature", bm25, chunks, top_k=3)
        for i, r in enumerate(results):
            print(f"[{i+1}] Score: {r['score']} | {r['manual_name']} p.{r['page_number']}")
            print(f"     {r['text'][:150]}")