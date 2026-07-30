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
    print("\n🔄 Loading vector search components...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✅ Embedding model loaded: {EMBEDDING_MODEL}")

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(COLLECTION_NAME)
    print(f"  ✅ ChromaDB connected: {collection.count()} chunks")

    return model, collection

def vector_search(
    query      : str,
    collection : chromadb.Collection,
    model      : SentenceTransformer,
    top_k      : int = TOP_K,
    manual_filter: str = None
) -> list:
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    # Step 2: Optional filter — restrict to one manual
    where_filter = None
    if manual_filter and manual_filter != "All Manuals":
        where_filter = {"manual_name": {"$eq": manual_filter}}
        print(
            f"  🎯 Vector filter: '{manual_filter}'"
        )
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=['documents', 'metadatas', 'distances'],
        where=where_filter
    )

    formatted = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        similarity = round(1 - dist, 4)
        formatted.append({
            "text"         : doc,
            "manual_name"  : meta['manual_name'],
            "page_number"  : meta['page_number'],
            "section_title": meta['section_title'],
            "token_count"  : meta['token_count'],
            "score"        : similarity,
            "source"       : "vector"
        })

    return formatted

def print_results(query: str, results: list) -> None:
    print(f"\n{'='*58}")
    print(f"  Query  : '{query}'")
    print(f"  Results: {len(results)}")
    print(f"{'='*58}\n")

    for i, r in enumerate(results):
        print(f"  [{i+1}] Similarity : {r['score']}")
        print(f"       Manual     : {r['manual_name']}")
        print(f"       Page       : {r['page_number']}")
        print(f"       Section    : {r['section_title']}")
        print(f"       Text       : {r['text'][:200]}...")
        print()

if __name__ == "__main__":
    model, collection = load_vector_components()

    test_queries = [
        "hydraulic pump losing pressure",
        "motor overheating after long operation",
        "compressor vibration excessive noise"
    ]

    for query in test_queries:
        results = vector_search(
            query     = query,
            collection= collection,
            model     = model,
            top_k     = 3
        )
        print_results(query, results)
        print("\n" + "-"*58 + "\n")