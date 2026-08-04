"""
Phase 4 - FastAPI Main Application
Complete version with PDF Upload + Multilingual Support + Cloud Embeddings (Zero RAM).
"""

# ── Fix imports FIRST ────────────────────────────────────
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
import requests
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import (
    FastAPI, HTTPException,
    BackgroundTasks, UploadFile, File
)
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
    log_feedback, get_recent_queries,
    get_feedback_stats
)

# ════════════════════════════════════════════════════════
# Cloud Embeddings Wrapper (Zero RAM Fix)
# ════════════════════════════════════════════════════════
class CloudEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        token = os.getenv('HF_API_TOKEN')
        self.headers = {"Authorization": f"Bearer {token}"}

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        response = requests.post(
            self.api_url, 
            headers=self.headers, 
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        if response.status_code != 200:
            raise Exception(f"HuggingFace API Error: {response.text}")
        return response.json()
        
    def embed_documents(self, texts):
        return self.encode(texts)
        
    def embed_query(self, text):
        return self.encode(text)[0]

# ════════════════════════════════════════════════════════
# Global State
# ════════════════════════════════════════════════════════
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
    """Load all components once at startup."""
    print("\n🚀 Starting RAG API — loading components...")

    try:
        from bm25_search import load_bm25_index
        from vector_search import load_vector_components
        from reranker import get_reranker
        from llm_chain import load_llm, check_ollama_running

        # Load BM25
        bm25_path = BASE_DIR / "data/processed_chunks/bm25_index.pkl"
        if bm25_path.exists():
            state["bm25"], state["chunks"] = load_bm25_index()
            print(f"  ✅ BM25 loaded: {len(state['chunks'])} chunks")
        else:
            print("⚠️  BM25 index not found — run retrieval/bm25_search.py")

        # Load vector search collection (and override model with Cloud)
        _, state["collection"] = load_vector_components()
        state["model"] = CloudEmbeddings()
        print("  ✅ CloudEmbeddings loaded successfully (Zero RAM mode)")

        print("  ✅ Using lightweight RRF ranking (no re-ranker)")

        # Load LLM
        if check_ollama_running():
            state["llm"] = load_llm()
            print("✅ Ollama LLM connected")
        else:
            print("⚠️  Ollama not running (Using Groq API)")

        state["ready"] = True
        print("✅ All components loaded — API is ready\n")

    except Exception as e:
        print(f"⚠️  Startup warning: {e}")
        print(traceback.format_exc())

    yield
    print("\n🛑 API shutting down...")


# ════════════════════════════════════════════════════════
# FastAPI App
# ════════════════════════════════════════════════════════
app = FastAPI(
    title      ="Industrial Maintenance RAG API",
    description=(
        "AI-powered repair instruction system with "
        "multilingual support. Describe a machine fault "
        "in any language and get repair instructions."
    ),
    version    ="2.0.0",
    lifespan   =lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins =["*"],
    allow_methods =["*"],
    allow_headers =["*"]
)


# ════════════════════════════════════════════════════════
# Helper
# ════════════════════════════════════════════════════════
def get_manual_counts() -> dict:
    """Read all metadata from ChromaDB and count chunks per manual."""
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


# ════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if API and all components are running."""
    from llm_chain import check_ollama_running
    chroma_count = 0
    if state["collection"] is not None:
        chroma_count = state["collection"].count()
    return HealthResponse(
        status        ="ready" if state["ready"] else "loading",
        ollama_running=check_ollama_running(),
        chroma_chunks =chroma_count,
        model         =os.getenv("OLLAMA_MODEL", "mistral")
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest, background_tasks: BackgroundTasks):
    """Submit fault description and get repair instructions."""
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="System is still loading. Please retry.")

    if state["collection"] is None or state["collection"].count() == 0:
        raise HTTPException(status_code=503, detail="No manuals ingested. Upload a manual first.")

    if state["bm25"] is None:
        raise HTTPException(status_code=503, detail="BM25 index not loaded.")

    query_id = generate_query_id()

    try:
        from reranker import full_retrieval_pipeline
        from llm_chain import generate_repair_instructions
        from translator import translate_to_english, translate_from_english, get_language_name

        # ── Step 1: Detect language + translate to English ──
        original_query = request.query
        english_query, detected_lang = translate_to_english(original_query)
        lang_name = get_language_name(detected_lang)

        if detected_lang != 'en':
            print(f"\n  🌐 Language detected: {lang_name} ({detected_lang})")
            print(f"  🔄 Translated to English: {english_query}")
        else:
            print(f"\n  🌐 Language: English")

        # ── Step 2: Retrieve using English query ────────────
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

        # ── Step 3: Generate answer in English ──────────────
        result = generate_repair_instructions(
            query           =english_query,
            retrieved_chunks=retrieved,
            llm             =state["llm"]
        )

        # ── Step 4: Translate answer back to user language ──
        if detected_lang not in ['en', 'english']:
            print(f"  🔄 Translating answer to {lang_name}...")
            result["answer"] = translate_from_english(result["answer"], detected_lang)
            print(f"  ✅ Answer translated to {lang_name}")

        # ── Step 5: Build response ──────────────────────────
        sources = [SourceCitation(**s) for s in result["sources"]]

        response = QueryResponse(
            query_id          =query_id,
            query             =original_query,
            answer            =result["answer"],
            sources           =sources,
            confidence        =result["confidence"],
            model             =result["model"],
            chunks_used       =result["chunks_used"],
            response_time_s   =result["response_time_s"],
            timestamp         =datetime.now().isoformat(),
            detected_language =lang_name,
            original_query    =original_query
        )

        # ── Step 6: Log in background ───────────────────────
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
        print(f"❌ Query error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-manual", tags=["Manuals"])
async def upload_manual(file: UploadFile = File(...)):
    """Upload a PDF manual directly from the UI."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    if state["collection"] is None:
        raise HTTPException(status_code=503, detail="System not ready. Start API first.")
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded.")

    try:
        from pdf_extractor import extract_pdf
        from chunker import chunk_pages
        from bm25_search import build_bm25_index, save_bm25_index

        # ── Save uploaded PDF ────────────────────────────
        manuals_dir = BASE_DIR / "data/raw_manuals"
        manuals_dir.mkdir(parents=True, exist_ok=True)
        save_path = manuals_dir / file.filename

        with open(save_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        print(f"\n📤 Uploaded: {file.filename}")

        # ── Step 1: Extract text ─────────────────────────
        print("\n  Step 1: Extracting text from PDF...")
        pages = extract_pdf(str(save_path))

        if not pages:
            raise HTTPException(status_code=400, detail="No text extracted from this PDF.")
        print(f"  ✅ Extracted {len(pages)} pages")

        processed_dir = BASE_DIR / "data/processed_chunks"
        processed_dir.mkdir(parents=True, exist_ok=True)
        clean_stem  = save_path.stem.replace(" ", "_")
        output_file = processed_dir / f"{clean_stem}_extracted.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pages, f, indent=2, ensure_ascii=False)

        # ── Step 2: Chunk ────────────────────────────────
        print("\n  Step 2: Chunking pages...")
        new_chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)
        if not new_chunks:
            raise HTTPException(status_code=400, detail="Could not create chunks from PDF.")
        print(f"  ✅ Created {len(new_chunks)} chunks")

        # ── Step 3: Delete old + embed new ───────────────
        print("\n  Step 3: Generating embeddings...")
        try:
            existing       = state["collection"].get(include=["metadatas"])
            existing_ids   = set(existing['ids'])
            existing_metas = existing.get("metadatas", [])
            manual_name    = save_path.stem
            ids_to_delete  = []

            for idx, eid in enumerate(existing['ids']):
                meta = existing_metas[idx] if idx < len(existing_metas) else {}
                if meta.get("manual_name") == manual_name:
                    ids_to_delete.append(eid)

            if ids_to_delete:
                state["collection"].delete(ids=ids_to_delete)
                print(f"  🗑️  Removed {len(ids_to_delete)} old chunks for '{manual_name}'")
                existing_ids -= set(ids_to_delete)
        except Exception as del_err:
            print(f"  ⚠️  Cleanup warning: {del_err}")

        # Embed in batches
        batch_size  = 32
        added_count = 0

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            texts = [chunk['text'] for chunk in batch]

            embeddings = state["model"].encode(
                texts,
                show_progress_bar   =False,
                convert_to_numpy    =True,
                normalize_embeddings=True
            )

            ids   = [c['chunk_id'] for c in batch]
            docs  = texts
            
            # Safe list conversion for API response
            embs = [list(e) if hasattr(e, 'tolist') else e for e in embeddings]
            
            metas = [
                {
                    "manual_name"  : c['manual_name'],
                    "page_number"  : c['page_number'],
                    "total_pages"  : c['total_pages'],
                    "section_title": c['section_title'],
                    "chunk_index"  : c['chunk_index'],
                    "token_count"  : c['token_count'],
                    "char_count"   : c['char_count']
                }
                for c in batch
            ]

            state["collection"].add(
                ids       =ids,
                documents =docs,
                embeddings=embs,
                metadatas =metas
            )
            added_count += len(ids)

        total_now = state["collection"].count()
        print(f"  ✅ Added {added_count} chunks to ChromaDB")

        # ── Step 4: Rebuild BM25 ─────────────────────────
        print("\n  Step 4: Rebuilding BM25 index...")
        all_chunks_file = processed_dir / "all_chunks.json"
        existing_chunks = []

        if all_chunks_file.exists():
            with open(all_chunks_file, 'r', encoding='utf-8') as f:
                existing_chunks = json.load(f)

        manual_name     = save_path.stem
        existing_chunks = [c for c in existing_chunks if c.get('manual_name') != manual_name]
        all_chunks = existing_chunks + new_chunks

        with open(all_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        new_bm25, updated_chunks = build_bm25_index(all_chunks)
        save_bm25_index(new_bm25, updated_chunks)

        state["bm25"]   = new_bm25
        state["chunks"] = updated_chunks

        updated_counts = get_manual_counts()
        manual_list    = list(updated_counts.keys())

        return {
            "status"      : "success",
            "filename"    : file.filename,
            "pages"       : len(pages),
            "chunks_added": added_count,
            "total_chunks": total_now,
            "manual_names": manual_list,
            "message"     : f"✅ Successfully ingested {file.filename}. {added_count} chunks added."
        }

    except HTTPException:
        raise
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"\n❌ Upload failed:\n{error_detail}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/manuals", response_model=ManualsResponse, tags=["Manuals"])
async def list_manuals():
    """List all ingested manuals."""
    if state["collection"] is None:
        raise HTTPException(status_code=503, detail="Vector database not initialized")
    try:
        counts = get_manual_counts()
        manuals = [
            ManualInfo(manual_name=name, chunk_count=count)
            for name, count in sorted(counts.items())
        ]
        return ManualsResponse(
            total_manuals=len(manuals),
            total_chunks =sum(m.chunk_count for m in manuals),
            manuals      =manuals
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading manuals: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """Submit rating for a query response."""
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
        message =f"Thank you for rating {request.rating}/5"
    )

@app.get("/history", tags=["System"])
async def query_history(limit: int = 10):
    """View recent query history."""
    queries = get_recent_queries(limit=limit)
    stats   = get_feedback_stats()
    return {"recent_queries": queries, "feedback_stats": stats}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=False)