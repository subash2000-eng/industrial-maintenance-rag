import os
from pathlib import Path
import chromadb

# Base directory setup
BASE_DIR = Path(__file__).parent.parent
DB_DIR = BASE_DIR / "data/chroma_db"

def load_vector_components():
    """
    Loads ONLY the ChromaDB collection.
    The model is now handled by CloudEmbeddings in main.py to save RAM.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB Client
    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    # Get or create the collection
    collection = client.get_or_create_collection(
        name="manuals_collection",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Return None for the model (handled in main.py) and the collection
    return None, collection