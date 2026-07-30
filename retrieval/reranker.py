"""
Phase 2 - Cross-Encoder Re-Ranker
Takes top-K hybrid results and re-scores them
by reading query + document together.
Much more precise than embedding similarity alone.
"""

# ── Fix imports FIRST ────────────────────────────────────
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "retrieval"))
sys.path.insert(0, str(BASE_DIR / "ingestion"))

import os
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────
TOP_K_RERANK   = int(os.getenv("TOP_K_RERANK", 3))
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load model once — reuse every query
_reranker_model = None


def get_reranker() -> CrossEncoder:
    """
    Load cross-encoder model.
    Cached so it only loads once per session.
    First run downloads ~80MB automatically.
    """
    global _reranker_model

    if _reranker_model is None:
        print(f"\n🔄 Loading re-ranker model...")
        print(f"   {RERANKER_MODEL}")
        print(f"   ℹ️  First run downloads ~80MB automatically")
        _reranker_model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512
        )
        print(f"  ✅ Re-ranker loaded")

    return _reranker_model


def rerank(
    query     : str,
    candidates: list,
    top_k     : int = TOP_K_RERANK,
    reranker  : CrossEncoder = None
) -> list:
    """
    Re-rank candidate chunks using cross-encoder.

    Cross-encoder reads (query + chunk) together
    and gives a precise relevance score to each pair.

    Args:
        query     : Original user query
        candidates: Top-K results from hybrid retriever
        top_k     : How many to keep after re-ranking
        reranker  : CrossEncoder model

    Returns:
        Top-k re-ranked results with rerank_score added
    """
    if not candidates:
        return []

    if reranker is None:
        reranker = get_reranker()

    # Build (query, chunk_text) pairs
    # Cross-encoder reads each pair together
    pairs = [
        (query, candidate['text'])
        for candidate in candidates
    ]

    # Score all pairs at once
    scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )

    # Attach rerank score to each candidate
    scored = []
    for candidate, score in zip(candidates, scores):
        result                 = candidate.copy()
        result['rerank_score'] = round(float(score), 4)
        scored.append(result)

    # Sort highest score first
    scored.sort(
        key=lambda x: x['rerank_score'],
        reverse=True
    )

    return scored[:top_k]


def compute_confidence(rerank_score: float) -> str:
    """
    Convert rerank score to human readable confidence label.
    """
    if rerank_score > 5:
        return "Very High"
    elif rerank_score > 2:
        return "High"
    elif rerank_score > 0:
        return "Medium"
    elif rerank_score > -2:
        return "Low"
    else:
        return "Very Low"


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
    Complete retrieval pipeline in one function call:

    Query
      → Hybrid Search (BM25 + Vector + RRF) → top 20
      → Cross-Encoder Re-rank               → final 3

    Args:
        query             : Technician's fault description
        bm25              : BM25Okapi model
        chunks            : All chunks list
        vector_collection : ChromaDB collection
        embedding_model   : SentenceTransformer model
        top_k_retrieve    : Chunks before reranking
        top_k_rerank      : Final chunks after reranking

    Returns:
        Final top-k results ready for LLM generation
    """
    from hybrid_retriever import hybrid_retrieve

    # Step 1: Hybrid retrieval — get top 20
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

    # Step 2: Re-rank — pick best 3
    print(f"\n🎯 Re-ranking to find best {top_k_rerank}...")
    reranker      = get_reranker()
    final_results = rerank(
        query     =query,
        candidates=candidates,
        top_k     =top_k_rerank,
        reranker  =reranker
    )

    # Step 3: Add confidence labels
    for result in final_results:
        result['confidence'] = compute_confidence(
            result['rerank_score']
        )

    print(f"  ✅ Final {len(final_results)} chunks selected")
    return final_results


# ── Run this file directly to test ──────────────────────
if __name__ == "__main__":

    from bm25_search   import load_bm25_index
    from vector_search import load_vector_components

    print("🎯 Testing Full Retrieval Pipeline...\n")

    try:
        # Load all components
        bm25, chunks      = load_bm25_index()
        model, collection = load_vector_components()

        # Test queries
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
                print(f"       Section   : {r['section_title']}")
                print(f"       Text      : {r['text'][:200]}...")
                print()

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Run ingest_pipeline.py and bm25_search.py first")