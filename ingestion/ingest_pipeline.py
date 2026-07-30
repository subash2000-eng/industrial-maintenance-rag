import os
import time
from pathlib import Path
from dotenv import load_dotenv

from pdf_extractor import extract_all_pdfs, get_extraction_stats
from chunker       import chunk_pages, save_chunks, get_chunking_stats
from embedder      import (
    load_embedding_model,
    get_chroma_client,
    get_or_create_collection,
    embed_and_store_chunks,
    verify_embedding
)

load_dotenv()

BASE_DIR        = Path(__file__).parent.parent
MANUALS_DIR     = BASE_DIR / "data/raw_manuals"
PROCESSED_DIR   = BASE_DIR / "data/processed_chunks"
CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "data/chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "maintenance_manuals")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 50))

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║     Industrial Maintenance Manual RAG — Ingestion        ║
║     PDF  →  Chunks  →  Embeddings  →  ChromaDB           ║
╚══════════════════════════════════════════════════════════╝
    """)

def check_manuals_exist() -> bool:
    pdf_files = list(MANUALS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n⚠️  No PDF files found in: {MANUALS_DIR}")
        print("Please add industrial maintenance manual PDFs to that folder.\nThen run this script again.\n")
        return False

    print(f"✅ Found {len(pdf_files)} PDF manual(s):")
    for f in pdf_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   • {f.name} ({size_mb:.1f} MB)")
    return True

def run_full_pipeline():
    print_banner()
    start_time = time.time()

    MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not check_manuals_exist():
        return

    print("\n" + "=" * 58)
    print("  STEP 1/4 — Extracting text from PDFs")
    print("=" * 58)

    pages = extract_all_pdfs(
        manuals_dir=str(MANUALS_DIR),
        output_dir=str(PROCESSED_DIR)
    )

    if not pages:
        print("❌ No content extracted. Check your PDF files.")
        return

    ext_stats = get_extraction_stats(pages)
    print(f"\n  📊 {ext_stats['total_pages']} pages from {ext_stats['total_manuals']} manual(s)")

    print("\n" + "=" * 58)
    print("  STEP 2/4 — Chunking extracted text")
    print("=" * 58)

    chunks = chunk_pages(
        pages=pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    if not chunks:
        print("❌ No chunks created.")
        return

    chunks_file = PROCESSED_DIR / "all_chunks.json"
    save_chunks(chunks, str(chunks_file))

    chunk_stats = get_chunking_stats(chunks)
    print(f"\n  📊 {chunk_stats['total_chunks']} chunks created | avg {chunk_stats['avg_tokens_per_chunk']} tokens each")

    print("\n" + "=" * 58)
    print("  STEP 3/4 — Generating embeddings")
    print("=" * 58)
    print(f"  Model: {EMBEDDING_MODEL}")
    print("  ℹ️  First run downloads model (~90MB) automatically")

    model      = load_embedding_model(EMBEDDING_MODEL)
    client     = get_chroma_client(CHROMA_DB_PATH)
    collection = get_or_create_collection(client, COLLECTION_NAME)

    embed_and_store_chunks(chunks, collection, model)

    print("\n" + "=" * 58)
    print("  STEP 4/4 — Verifying retrieval")
    print("=" * 58)

    verify_embedding(
        collection,
        model,
        "bearing overheating fault diagnosis"
    )

    elapsed = round(time.time() - start_time, 1)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║                ✅  INGESTION COMPLETE                    ║
╠══════════════════════════════════════════════════════════╣
║  Manuals processed  : {ext_stats['total_manuals']:<33} ║
║  Pages extracted    : {ext_stats['total_pages']:<33} ║
║  Chunks created     : {chunk_stats['total_chunks']:<33} ║
║  Embeddings stored  : {collection.count():<33} ║
║  Time taken         : {elapsed}s{'':<31} ║
╠══════════════════════════════════════════════════════════╣
║  ▶  Next: Build Phase 2 — Retrieval Pipeline             ║
╚══════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    run_full_pipeline()