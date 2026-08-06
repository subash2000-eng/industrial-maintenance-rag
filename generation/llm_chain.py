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

from prompt_templates import build_rag_prompt, build_no_context_response, SYSTEM_PROMPT

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def check_ollama_running() -> bool:
    """Checks if Groq API key is configured."""
    return bool(GROQ_API_KEY)


def load_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in .env file.")
    llm = ChatGroq(
        temperature=0.1,
        model_name =GROQ_MODEL,
        api_key    =GROQ_API_KEY
    )
    print(f"LLM loaded: {GROQ_MODEL}")
    return llm


def generate_repair_instructions(
    query           : str,
    retrieved_chunks: list,
    llm             : ChatGroq = None
) -> dict:
    start_time = time.time()

    if not retrieved_chunks:
        return {
            "answer"         : build_no_context_response(),
            "sources"        : [],
            "confidence"     : "No Context",
            "model"          : GROQ_MODEL,
            "response_time_s": 0,
            "chunks_used"    : 0
        }

    if not GROQ_API_KEY:
        return {
            "answer"         : "GROQ_API_KEY is missing in .env file.",
            "sources"        : [],
            "confidence"     : "Error",
            "model"          : GROQ_MODEL,
            "response_time_s": 0,
            "chunks_used"    : 0
        }

    if llm is None:
        llm = load_llm()

    prompt = build_rag_prompt(query, retrieved_chunks)

    try:
        response = llm.invoke(prompt)
        answer   = response.content.strip()
    except Exception as e:
        answer = f"Generation error: {str(e)}"

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

    elapsed = round(time.time() - start_time, 2)
    overall = retrieved_chunks[0].get('confidence', 'Unknown') if retrieved_chunks else 'Unknown'

    print(f"Response generated in {elapsed}s")

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
    if not retrieved_chunks:
        yield build_no_context_response()
        return

    if not GROQ_API_KEY:
        yield "GROQ_API_KEY is missing in .env file."
        return

    if llm is None:
        llm = load_llm()

    prompt = build_rag_prompt(query, retrieved_chunks)
    for chunk in llm.stream(prompt):
        yield chunk.content