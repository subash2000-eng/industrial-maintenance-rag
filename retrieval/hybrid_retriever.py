import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "retrieval"))
sys.path.insert(0, str(BASE_DIR / "ingestion"))

import os
from dotenv import load_dotenv

load_dotenv()

TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 20))
RRF_K = 60


def reciprocal_rank_fusion(
    bm25_results  : list,
    vector_results: list,
    k             : int = RRF_K
) -> list:
    scores = {}
    docs   = {}

    for rank, result in enumerate(bm25_results):
        text             = result['text']
        rrf_score        = 1 / (k + rank + 1)
        scores[text]     = scores.get(text, 0) + rrf_score
        if text not in docs:
            docs[text]            = result.copy()
            docs[text]['sources'] = ['bm25']
        else:
            docs[text]['sources'].append('bm25')

    for rank, result in enumerate(vector_results):
        text         = result['text']
        rrf_score    = 1 / (k + rank + 1)
        scores[text] = scores.get(text, 0) + rrf_score
        if text not in docs:
            docs[text]            = result.copy()
            docs[text]['sources'] = ['vector']
        else:
            if 'sources' not in docs[text]:
                docs[text]['sources'] = []
            docs[text]['sources'].append('vector')

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    merged = []
    for text, rrf_score in ranked:
        result              = docs[text].copy()
        result['rrf_score'] = round(rrf_score, 6)
        result['in_both']   = len(result.get('sources', [])) > 1
        merged.append(result)

    return merged


def hybrid_retrieve(
    query            : str,
    bm25             ,
    chunks           : list,
    vector_collection,
    embedding_model  ,
    top_k            : int = TOP_K,
    manual_filter    : str = None
) -> list:
    from bm25_search   import bm25_search
    from vector_search import vector_search

    filtered_chunks = chunks
    if manual_filter and manual_filter != "All Manuals":
        filtered_chunks = [
            c for c in chunks
            if c.get('manual_name') == manual_filter
        ]
        if not filtered_chunks:
            return []

    bm25_results   = bm25_search(query, bm25, filtered_chunks, top_k=top_k)
    vector_results = vector_search(
        query, vector_collection, embedding_model,
        top_k=top_k, manual_filter=manual_filter
    )

    merged = reciprocal_rank_fusion(bm25_results, vector_results)
    return merged[:top_k]