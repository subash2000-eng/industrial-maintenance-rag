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
    """Returns None. Re-ranker removed to reduce memory usage."""
    return None


def rerank(
    query     : str,
    candidates: list,
    top_k     : int = TOP_K_RERANK,
    reranker  = None
) -> list:
    """Select top results using RRF score."""
    if not candidates:
        return []

    sorted_candidates = sorted(
        candidates,
        key=lambda x: x.get('rrf_score', 0),
        reverse=True
    )

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
    from hybrid_retriever import hybrid_retrieve

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
        return []

    return rerank(query=query, candidates=candidates, top_k=top_k_rerank)