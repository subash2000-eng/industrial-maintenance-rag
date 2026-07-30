"""
Phase 2 - Hybrid Retriever
Combines BM25 + Vector Search using
Reciprocal Rank Fusion (RRF).
Fixed with manual_filter support.
"""

# ── Fix imports FIRST ────────────────────────────────────
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
    """
    Merge BM25 and vector results using RRF.
    Chunks appearing in both lists rank highest.
    """
    scores = {}
    docs   = {}

    # Score BM25 results
    for rank, result in enumerate(bm25_results):
        text      = result['text']
        rrf_score = 1 / (k + rank + 1)
        scores[text] = scores.get(text, 0) + rrf_score
        if text not in docs:
            docs[text]            = result.copy()
            docs[text]['sources'] = ['bm25']
        else:
            docs[text]['sources'].append('bm25')

    # Score vector results
    for rank, result in enumerate(vector_results):
        text      = result['text']
        rrf_score = 1 / (k + rank + 1)
        scores[text] = scores.get(text, 0) + rrf_score
        if text not in docs:
            docs[text]            = result.copy()
            docs[text]['sources'] = ['vector']
        else:
            if 'sources' not in docs[text]:
                docs[text]['sources'] = []
            docs[text]['sources'].append('vector')

    # Sort by combined RRF score
    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Build final list
    merged = []
    for text, rrf_score in ranked:
        result              = docs[text].copy()
        result['rrf_score'] = round(rrf_score, 6)
        result['in_both']   = len(
            result.get('sources', [])
        ) > 1
        merged.append(result)

    return merged


def hybrid_retrieve(
    query             : str,
    bm25              ,
    chunks            : list,
    vector_collection ,
    embedding_model   ,
    top_k             : int = TOP_K,
    manual_filter     : str = None
) -> list:
    """
    Run BM25 + Vector search and merge with RRF.
    Supports filtering by specific manual name.
    """
    from bm25_search   import bm25_search
    from vector_search import vector_search

    # ── Filter chunks by manual if specified ─────────────
    filtered_chunks = chunks
    if manual_filter and manual_filter != "All Manuals":
        filtered_chunks = [
            c for c in chunks
            if c.get('manual_name') == manual_filter
        ]
        print(
            f"  🎯 Manual filter: '{manual_filter}' "
            f"→ {len(filtered_chunks)} chunks"
        )

        # If no chunks found for this manual warn user
        if not filtered_chunks:
            print(
                f"  ⚠️  No chunks found for '{manual_filter}'"
            )
            print(
                f"  ℹ️  Available manuals: "
                f"{list(set(c['manual_name'] for c in chunks))}"
            )
            return []

    # ── BM25 keyword search ───────────────────────────────
    bm25_results = bm25_search(
        query,
        bm25,
        filtered_chunks,
        top_k=top_k
    )

    # ── Vector semantic search ────────────────────────────
    vector_results = vector_search(
        query,
        vector_collection,
        embedding_model,
        top_k        =top_k,
        manual_filter=manual_filter
    )

    # ── Merge with RRF ────────────────────────────────────
    merged = reciprocal_rank_fusion(
        bm25_results,
        vector_results
    )

    return merged[:top_k]


def format_retrieval_summary(results: list) -> str:
    """Print results clearly."""
    lines = [
        f"\n📋 Hybrid Retrieval — {len(results)} results\n"
    ]

    for i, r in enumerate(results):
        sources = ' + '.join(r.get('sources', ['unknown']))
        star    = "⭐ " if r.get('in_both') else "   "
        lines.append(
            f"{star}[{i+1}]"
            f"  RRF: {r.get('rrf_score', 0):.5f}"
            f"  [{sources}]"
            f"  {r['manual_name']}"
            f"  p.{r['page_number']}"
        )
        lines.append(f"       Section: {r['section_title']}")
        lines.append(f"       Text   : {r['text'][:150]}...")
        lines.append("")

    return '\n'.join(lines)


# ── Run this file directly to test ──────────────────────
if __name__ == "__main__":

    from bm25_search   import load_bm25_index
    from vector_search import load_vector_components

    print("🔀 Testing Hybrid Retriever...\n")

    try:
        bm25, chunks      = load_bm25_index()
        model, collection = load_vector_components()

        test_queries = [
            "compressor vibration excessive noise",
            "motor overheating temperature protection",
            "pressure relief valve maintenance"
        ]

        for query in test_queries:
            results = hybrid_retrieve(
                query             =query,
                bm25              =bm25,
                chunks            =chunks,
                vector_collection =collection,
                embedding_model   =model,
                top_k             =10
            )

            print(f"Query: '{query}'")
            print(format_retrieval_summary(results))
            print("─" * 58 + "\n")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(
            "   Run ingest_pipeline.py and "
            "bm25_search.py first"
        )