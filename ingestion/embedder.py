import json
import os
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH",  "./data/chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "maintenance_manuals")
BATCH_SIZE      = 64


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimensions: {model.get_sentence_embedding_dimension()}")
    return model


def get_chroma_client(db_path: str = CHROMA_DB_PATH) -> chromadb.PersistentClient:
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path    =str(db_path),
        settings=Settings(anonymized_telemetry=False)
    )


def get_or_create_collection(
    client         : chromadb.PersistentClient,
    collection_name: str = COLLECTION_NAME
) -> chromadb.Collection:
    collection = client.get_or_create_collection(
        name    =collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection: {collection_name} | Documents: {collection.count()}")
    return collection


def embed_and_store_chunks(
    chunks    : list,
    collection: chromadb.Collection,
    model     : SentenceTransformer,
    batch_size: int = BATCH_SIZE
) -> None:
    existing     = collection.get(include=[])
    existing_ids = set(existing['ids'])
    new_chunks   = [c for c in chunks if c['chunk_id'] not in existing_ids]

    if not new_chunks:
        print("All chunks already embedded.")
        return

    print(f"Embedding {len(new_chunks)} new chunks...")

    for i in tqdm(range(0, len(new_chunks), batch_size), desc="Embedding"):
        batch      = new_chunks[i : i + batch_size]
        texts      = [chunk['text'] for chunk in batch]
        embeddings = model.encode(
            texts,
            show_progress_bar   =False,
            convert_to_numpy    =True,
            normalize_embeddings=True
        )

        collection.add(
            ids       =[chunk['chunk_id'] for chunk in batch],
            documents =texts,
            embeddings=embeddings.tolist(),
            metadatas =[
                {
                    "manual_name"  : chunk['manual_name'],
                    "page_number"  : chunk['page_number'],
                    "total_pages"  : chunk['total_pages'],
                    "section_title": chunk['section_title'],
                    "chunk_index"  : chunk['chunk_index'],
                    "token_count"  : chunk['token_count'],
                    "char_count"   : chunk['char_count']
                }
                for chunk in batch
            ]
        )

    print(f"ChromaDB now contains {collection.count()} chunks")


if __name__ == "__main__":
    BASE_DIR    = Path(__file__).parent.parent
    chunks_file = BASE_DIR / "data/processed_chunks/all_chunks.json"

    if not chunks_file.exists():
        print("Run chunker.py first.")
    else:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        model      = load_embedding_model()
        client     = get_chroma_client(str(BASE_DIR / "data/chroma_db"))
        collection = get_or_create_collection(client)
        embed_and_store_chunks(chunks, collection, model)