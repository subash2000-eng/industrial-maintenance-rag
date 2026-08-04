"""
Phase 2 - Retrieval Pipeline (Lightweight Version)
Removed cross-encoder re-ranker to save RAM.
Uses hybrid search (BM25 + Vector + RRF) only.
Works perfectly on Render free tier (512MB RAM).
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "retrieval"))
sys.path.insert(0, str(BASE_DIR / "ingestion"))

import os
from dotenv import load_dotenv

load_dotenv()

TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 3))


def get_reranker():
    """
    Reranker removed to save RAM.
    Returns None — kept for API compatibility.
    """
    return None


def rerank(
    query     : str,
    candidates: list,
    top_k     : int = TOP_K_RERANK,
    reranker  = None
) -> list:
    """
    Instead of cross-encoder re-ranking,
    use RRF score to select top results.
    RRF already combines BM25 + Vector rankings.
    """
    if not candidates:
        return []

    # Sort by RRF score — already combined BM25 + Vector
    sorted_candidates = sorted(
        candidates,
        key=lambda x: x.get('rrf_score', 0),
        reverse=True
    )

    # Add confidence based on RRF score
    for result in sorted_candidates[:top_k]:
        rrf = result.get('rrf_score', 0)
        if rrf > 0.03:
            result['rerank_score'] = 8.0
            result['confidence']   = "High"
        elif rrf > 0.02:
            result['rerank_score'] = 5.0
            result['confidence']   = "Medium"
        else:
            result['rerank_score'] = 2.0
            result['confidence']   = "Low"

    return sorted_candidates[:top_k]


def compute_confidence(rerank_score: float) -> str:
    """Convert score to confidence label."""
    if rerank_score > 5:
        return "High"
    elif rerank_score > 2:
        return "Medium"
    else:
        return "Low"


def full_retrieval_pipeline(
    query             : str,
    bm25              ,
    chunks            : list,
    vector_collection ,
    embedding_model   ,
    top_k_retrieve    : int = 20,
    top_k_rerank      : int = TOP_K_RERANK,
    manual_filter     : str = None
) -> list:
    """
    Lightweight retrieval pipeline:
    BM25 + Vector Search + RRF → top results
    No cross-encoder (saves 500MB RAM)
    """
    from hybrid_retriever import hybrid_retrieve

    # Step 1: Hybrid retrieval
    print(f"\n🔍 Hybrid search: top {top_k_retrieve}...")
    candidates = hybrid_retrieve(
        query             =query,
        bm25              =bm25,
        chunks            =chunks,
        vector_collection =vector_collection,
        embedding_model   =embedding_model,
        top_k             =top_k_retrieve,
        manual_filter     =manual_filter
    )

    if not candidates:
        print("  ⚠️  No candidates found")
        return []

    print(f"  ✅ {len(candidates)} candidates retrieved")

    # Step 2: Select top results by RRF score
    print(f"\n🎯 Selecting best {top_k_rerank} by RRF score...")
    final_results = rerank(
        query     =query,
        candidates=candidates,
        top_k     =top_k_rerank
    )

    print(f"  ✅ Final {len(final_results)} chunks selected")
    return final_results


# ── Run this file directly to test ──────────────────────
if __name__ == "__main__":

    from bm25_search   import load_bm25_index
    from vector_search import load_vector_components

    print("🎯 Testing Lightweight Retrieval Pipeline...\n")

    try:
        bm25, chunks      = load_bm25_index()
        model, collection = load_vector_components()

        test_queries = [
            "motor bearing overheating symptoms",
            "compressor pressure drop fault",
            "oil seal leaking hydraulic system"
        ]

        for query in test_queries:
            print(f"\n{'='*58}")
            print(f"  Query: '{query}'")
            print(f"{'='*58}")

            results = full_retrieval_pipeline(
                query             =query,
                bm25              =bm25,
                chunks            =chunks,
                vector_collection =collection,
                embedding_model   =model,
                top_k_retrieve    =20,
                top_k_rerank      =3
            )

            print(f"\n🏆 Final {len(results)} chunks:\n")
            for i, r in enumerate(results):
                print(f"  [{i+1}] Score     : {r['rerank_score']}")
                print(f"       Confidence: {r['confidence']}")
                print(f"       Manual    : {r['manual_name']}")
                print(f"       Page      : {r['page_number']}")
                print(f"       Text      : {r['text'][:200]}...")
                print()

    except FileNotFoundError as e:
        print(f"❌ {e}")