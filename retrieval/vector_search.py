import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "maintenance_manuals")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K           = int(os.getenv("TOP_K_RETRIEVAL", 20))


def load_vector_components():
    print("Loading vector search components...")
    model  = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(
        path    =CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(COLLECTION_NAME)
    print(f"ChromaDB connected: {collection.count()} chunks")
    return model, collection


def vector_search(
    query        : str,
    collection   : chromadb.Collection,
    model        : SentenceTransformer,
    top_k        : int = TOP_K,
    manual_filter: str = None
) -> list:
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    where_filter = None
    if manual_filter and manual_filter != "All Manuals":
        where_filter = {"manual_name": {"$eq": manual_filter}}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results       =min(top_k, collection.count()),
        include         =['documents', 'metadatas', 'distances'],
        where           =where_filter
    )

    formatted = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        formatted.append({
            "text"         : doc,
            "manual_name"  : meta['manual_name'],
            "page_number"  : meta['page_number'],
            "section_title": meta['section_title'],
            "token_count"  : meta['token_count'],
            "score"        : round(1 - dist, 4),
            "source"       : "vector"
        })

    return formatted