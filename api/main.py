import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "retrieval"))
sys.path.insert(0, str(BASE_DIR / "ingestion"))
sys.path.insert(0, str(BASE_DIR / "generation"))
sys.path.insert(0, str(BASE_DIR / "api"))

import os
import json
import traceback
from datetime       import datetime
from contextlib     import asynccontextmanager
from dotenv         import load_dotenv

from fastapi             import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(BASE_DIR / ".env")

from schemas import (
    QueryRequest, QueryResponse,
    FeedbackRequest, FeedbackResponse,
    ManualsResponse, ManualInfo,
    HealthResponse, SourceCitation
)
from logger import (
    generate_query_id, log_query,
    log_feedback, get_recent_queries, get_feedback_stats
)

state = {
    "bm25"      : None,
    "chunks"    : None,
    "model"     : None,
    "collection": None,
    "llm"       : None,
    "ready"     : False
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting RAG API...")

    try:
        from bm25_search   import load_bm25_index
        from vector_search import load_vector_components
        from llm_chain     import load_llm, check_ollama_running

        bm25_path = BASE_DIR / "data/processed_chunks/bm25_index.pkl"
        if bm25_path.exists():
            state["bm25"], state["chunks"] = load_bm25_index()
        else:
            print("BM25 index not found. Run retrieval/bm25_search.py first.")

        state["model"], state["collection"] = load_vector_components()

        if check_ollama_running():
            state["llm"] = load_llm()
        else:
            print("GROQ_API_KEY not set. Add it to .env file.")

        state["ready"] = True
        print("API ready.")

    except Exception as e:
        print(f"Startup error: {e}")
        print(traceback.format_exc())

    yield

    print("API shutting down.")


app = FastAPI(
    title      ="Industrial Maintenance RAG API",
    description="AI-powered repair instruction system with multilingual support.",
    version    ="1.0.0",
    lifespan   =lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins =["*"],
    allow_methods =["*"],
    allow_headers =["*"]
)


def get_manual_counts() -> dict:
    if state["collection"] is None:
        return {}
    try:
        all_data  = state["collection"].get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        counts    = {}
        for meta in metadatas:
            name         = meta.get("manual_name", "unknown")
            counts[name] = counts.get(name, 0) + 1
        return counts
    except Exception:
        return {}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    from llm_chain import check_ollama_running
    chroma_count = state["collection"].count() if state["collection"] else 0
    return HealthResponse(
        status        ="ready" if state["ready"] else "loading",
        ollama_running=check_ollama_running(),
        chroma_chunks =chroma_count,
        model         =os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest, background_tasks: BackgroundTasks):
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="System is still loading.")

    if state["collection"] is None or state["collection"].count() == 0:
        raise HTTPException(status_code=503, detail="No manuals ingested.")

    if state["bm25"] is None:
        raise HTTPException(status_code=503, detail="BM25 index not loaded.")

    query_id = generate_query_id()

    try:
        from reranker   import full_retrieval_pipeline
        from llm_chain  import generate_repair_instructions
        from translator import translate_to_english, translate_from_english, get_language_name

        original_query               = request.query
        english_query, detected_lang = translate_to_english(original_query)
        lang_name                    = get_language_name(detected_lang)

        retrieved = full_retrieval_pipeline(
            query             =english_query,
            bm25              =state["bm25"],
            chunks            =state["chunks"],
            vector_collection =state["collection"],
            embedding_model   =state["model"],
            top_k_retrieve    =request.top_k_retrieve,
            top_k_rerank      =request.top_k_rerank,
            manual_filter     =request.manual_filter
        )

        result = generate_repair_instructions(
            query           =english_query,
            retrieved_chunks=retrieved,
            llm             =state["llm"]
        )

        if detected_lang not in ['en', 'english']:
            result["answer"] = translate_from_english(result["answer"], detected_lang)

        sources  = [SourceCitation(**s) for s in result["sources"]]
        response = QueryResponse(
            query_id         =query_id,
            query            =original_query,
            answer           =result["answer"],
            sources          =sources,
            confidence       =result["confidence"],
            model            =result["model"],
            chunks_used      =result["chunks_used"],
            response_time_s  =result["response_time_s"],
            timestamp        =datetime.now().isoformat(),
            detected_language=lang_name,
            original_query   =original_query
        )

        background_tasks.add_task(
            log_query,
            query_id        =query_id,
            query           =original_query,
            answer          =result["answer"],
            confidence      =result["confidence"],
            chunks_used     =result["chunks_used"],
            response_time_s =result["response_time_s"],
            model           =result["model"],
            sources         =result["sources"]
        )

        return response

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-manual", tags=["Manuals"])
async def upload_manual(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    if state["collection"] is None or state["model"] is None:
        raise HTTPException(status_code=503, detail="System not ready.")

    try:
        from pdf_extractor import extract_pdf
        from chunker       import chunk_pages
        from bm25_search   import build_bm25_index, save_bm25_index

        manuals_dir = BASE_DIR / "data/raw_manuals"
        manuals_dir.mkdir(parents=True, exist_ok=True)
        save_path = manuals_dir / file.filename

        with open(save_path, 'wb') as f:
            f.write(await file.read())

        pages = extract_pdf(str(save_path))
        if not pages:
            raise HTTPException(status_code=400, detail="No text extracted from PDF.")

        processed_dir = BASE_DIR / "data/processed_chunks"
        processed_dir.mkdir(parents=True, exist_ok=True)

        with open(processed_dir / f"{save_path.stem}_extracted.json", 'w', encoding='utf-8') as f:
            json.dump(pages, f, indent=2, ensure_ascii=False)

        new_chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)
        if not new_chunks:
            raise HTTPException(status_code=400, detail="Could not create chunks.")

        # Remove old chunks for this manual then re-embed
        try:
            existing       = state["collection"].get(include=["metadatas"])
            ids_to_delete  = [
                eid for eid, meta in zip(existing['ids'], existing.get('metadatas', []))
                if meta.get("manual_name") == save_path.stem
            ]
            if ids_to_delete:
                state["collection"].delete(ids=ids_to_delete)
        except Exception:
            pass

        # Embed in batches
        batch_size  = 32
        added_count = 0
        for i in range(0, len(new_chunks), batch_size):
            batch      = new_chunks[i : i + batch_size]
            texts      = [c['text'] for c in batch]
            embeddings = state["model"].encode(
                texts, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True
            )
            state["collection"].add(
                ids       =[c['chunk_id'] for c in batch],
                documents =texts,
                embeddings=embeddings.tolist(),
                metadatas =[{
                    "manual_name"  : c['manual_name'],
                    "page_number"  : c['page_number'],
                    "total_pages"  : c['total_pages'],
                    "section_title": c['section_title'],
                    "chunk_index"  : c['chunk_index'],
                    "token_count"  : c['token_count'],
                    "char_count"   : c['char_count']
                } for c in batch]
            )
            added_count += len(batch)

        # Rebuild BM25
        all_chunks_file = processed_dir / "all_chunks.json"
        existing_chunks = []
        if all_chunks_file.exists():
            with open(all_chunks_file, 'r', encoding='utf-8') as f:
                existing_chunks = json.load(f)

        existing_chunks = [c for c in existing_chunks if c.get('manual_name') != save_path.stem]
        all_chunks      = existing_chunks + new_chunks

        with open(all_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        new_bm25, updated_chunks = build_bm25_index(all_chunks)
        save_bm25_index(new_bm25, updated_chunks)
        state["bm25"]   = new_bm25
        state["chunks"] = updated_chunks

        updated_counts = get_manual_counts()

        return {
            "status"      : "success",
            "filename"    : file.filename,
            "pages"       : len(pages),
            "chunks_added": added_count,
            "total_chunks": state["collection"].count(),
            "manual_names": list(updated_counts.keys()),
            "message"     : f"Successfully ingested {file.filename}."
        }

    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/manuals", response_model=ManualsResponse, tags=["Manuals"])
async def list_manuals():
    if state["collection"] is None:
        raise HTTPException(status_code=503, detail="Vector database not initialized.")

    counts  = get_manual_counts()
    manuals = [
        ManualInfo(manual_name=name, chunk_count=count)
        for name, count in sorted(counts.items())
    ]
    return ManualsResponse(
        total_manuals=len(manuals),
        total_chunks =sum(m.chunk_count for m in manuals),
        manuals      =manuals
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        log_feedback,
        query_id   =request.query_id,
        rating     =request.rating,
        was_helpful=request.was_helpful,
        comment    =request.comment
    )
    return FeedbackResponse(
        status  ="success",
        query_id=request.query_id,
        message =f"Thank you for your feedback."
    )


@app.get("/history", tags=["System"])
async def query_history(limit: int = 10):
    return {
        "recent_queries": get_recent_queries(limit=limit),
        "feedback_stats": get_feedback_stats()
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=False)