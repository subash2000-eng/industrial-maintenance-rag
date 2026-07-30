"""
Phase 3 - LLM Chain (Cloud Version)
Connects retrieved chunks to Groq API (Mixtral).
Generates structured repair instructions with citations.
"""

# ── Fix imports FIRST ────────────────────────────────────
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "retrieval"))
sys.path.insert(0, str(BASE_DIR / "ingestion"))
sys.path.insert(0, str(BASE_DIR / "generation"))

import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompt_templates import (
    build_rag_prompt,
    build_no_context_response,
    SYSTEM_PROMPT
)

load_dotenv()

# ── Config ───────────────────────────────────────────────
GROQ_MODEL = "llama-3.1-8b-instant" # Latest fast model from Groq

def check_ollama_running() -> bool:
    """
    Mock function name kept to avoid breaking main.py!
    Instead of checking local Ollama, it now checks if Groq API key exists.
    """
    return bool(os.getenv("GROQ_API_KEY"))


def load_llm() -> ChatGroq:
    """
    Load Mixtral LLM via Groq API.
    """
    print(f"\n🤖 Loading Cloud LLM: {GROQ_MODEL}")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing in .env file!")

    llm = ChatGroq(
        temperature=0.1,
        model_name=GROQ_MODEL,
        api_key=api_key
    )
    print(f"  ✅ LLM ready: {GROQ_MODEL}")
    return llm


def generate_repair_instructions(
    query           : str,
    retrieved_chunks: list,
    llm             : ChatGroq = None
) -> dict:
    
    start_time = time.time()

    # ── Handle no context case ───────────────────────────
    if not retrieved_chunks:
        return {
            "answer"         : build_no_context_response(),
            "sources"        : [],
            "confidence"     : "No Context",
            "model"          : GROQ_MODEL,
            "response_time_s": 0,
            "chunks_used"    : 0
        }

    # ── Check API Key ────────────────────────────────────
    if not check_ollama_running():
        return {
            "answer": "Groq API Key is missing in .env file! Please add GROQ_API_KEY.",
            "sources"        : [],
            "confidence"     : "Error",
            "model"          : GROQ_MODEL,
            "response_time_s": 0,
            "chunks_used"    : 0
        }

    # ── Load LLM if not provided ─────────────────────────
    if llm is None:
        llm = load_llm()

    # ── Build prompt ─────────────────────────────────────
    prompt = build_rag_prompt(query, retrieved_chunks)

    # ── Generate response ────────────────────────────────
    print(f"\n🤖 Generating response via Groq API...")
    print(f"   Chunks: {len(retrieved_chunks)}")

    try:
        # Groq returns an AIMessage, so we need .content
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"Generation error: {str(e)}"

    # ── Build source citations ───────────────────────────
    sources = []
    seen    = set()

    for i, chunk in enumerate(retrieved_chunks):
        key = f"{chunk['manual_name']}_{chunk['page_number']}"
        if key not in seen:
            sources.append({
                "source_num"   : i + 1,
                "manual_name"  : chunk['manual_name'],
                "page_number"  : chunk['page_number'],
                "section_title": chunk['section_title'],
                "confidence"   : chunk.get('confidence', 'Unknown'),
                "rerank_score" : chunk.get('rerank_score', 0)
            })
            seen.add(key)

    # ── Calculate timing ─────────────────────────────────
    elapsed = round(time.time() - start_time, 2)

    # ── Overall confidence from top chunk ────────────────
    overall = retrieved_chunks[0].get(
        'confidence', 'Unknown'
    ) if retrieved_chunks else 'Unknown'

    print(f"  ✅ Response generated in {elapsed}s")

    return {
        "answer"         : answer,
        "sources"        : sources,
        "confidence"     : overall,
        "model"          : GROQ_MODEL,
        "response_time_s": elapsed,
        "chunks_used"    : len(retrieved_chunks)
    }


def stream_repair_instructions(
    query           : str,
    retrieved_chunks: list,
    llm             : ChatGroq = None
):
    """
    Stream LLM response token by token via Groq.
    """
    if not retrieved_chunks:
        yield build_no_context_response()
        return

    if not check_ollama_running():
        yield "Groq API Key is missing in .env file! Please add GROQ_API_KEY."
        return

    if llm is None:
        llm = load_llm()

    prompt = build_rag_prompt(query, retrieved_chunks)

    # Stream chunks from Groq and extract content
    for chunk in llm.stream(prompt):
        yield chunk.content


# ── Run this file directly to test ──────────────────────
if __name__ == "__main__":

    from bm25_search   import load_bm25_index
    from vector_search import load_vector_components
    from reranker      import full_retrieval_pipeline

    print("🧪 Testing Cloud LLM Generation Pipeline...\n")

    if not check_ollama_running():
        print("❌ Groq API Key missing!")
        exit()

    try:
        bm25, chunks      = load_bm25_index()
        model, collection = load_vector_components()
        llm = load_llm()

        query = "motor bearing overheating after long operation"

        print(f"\n🔍 Retrieving relevant chunks...")
        retrieved = full_retrieval_pipeline(
            query             =query,
            bm25              =bm25,
            chunks            =chunks,
            vector_collection =collection,
            embedding_model   =model,
            top_k_retrieve    =20,
            top_k_rerank      =3
        )

        print("🤖 Generating repair instructions (streaming)...\n")
        for token in stream_repair_instructions(query, retrieved, llm):
            print(token, end='', flush=True)

        print(f"\n\n✅ Generation complete")

    except Exception as e:
        print(f"❌ Error: {e}")