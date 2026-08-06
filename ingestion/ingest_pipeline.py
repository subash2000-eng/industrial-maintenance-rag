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
    embed_and_store_chunks
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


def run_full_pipeline():
    start_time = time.time()

    MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(MANUALS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {MANUALS_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF manual(s)")

    # Step 1: Extract
    pages = extract_all_pdfs(
        manuals_dir=str(MANUALS_DIR),
        output_dir =str(PROCESSED_DIR)
    )
    if not pages:
        print("No content extracted.")
        return

    ext_stats = get_extraction_stats(pages)
    print(f"Extracted {ext_stats['total_pages']} pages from {ext_stats['total_manuals']} manual(s)")

    # Step 2: Chunk
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not chunks:
        print("No chunks created.")
        return

    save_chunks(chunks, str(PROCESSED_DIR / "all_chunks.json"))
    chunk_stats = get_chunking_stats(chunks)
    print(f"Created {chunk_stats['total_chunks']} chunks")

    # Step 3: Embed
    model      = load_embedding_model(EMBEDDING_MODEL)
    client     = get_chroma_client(CHROMA_DB_PATH)
    collection = get_or_create_collection(client, COLLECTION_NAME)
    embed_and_store_chunks(chunks, collection, model)

    elapsed = round(time.time() - start_time, 1)
    print(f"Ingestion complete in {elapsed}s")
    print(f"Total chunks in ChromaDB: {collection.count()}")


if __name__ == "__main__":
    run_full_pipeline()