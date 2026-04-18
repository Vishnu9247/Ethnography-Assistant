"""
cybertron.py — Vector Database
Manages a persistent ChromaDB collection named 'cybertron'.
"""

import chromadb
from langchain_ollama import OllamaEmbeddings
import uuid


CHROMA_PATH = "./cybertron_db"
COLLECTION_NAME = "cybertron"

_client = None
_collection = None
_embedder = None


def initialize_db():
    """Set up the ChromaDB client and collection with Ollama embeddings."""
    global _client, _collection, _embedder

    _embedder = OllamaEmbeddings(model="llama3.2")
    _client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"[Cybertron] Database initialized at '{CHROMA_PATH}' — collection '{COLLECTION_NAME}'")
    return _collection


def _get_collection():
    global _collection
    if _collection is None:
        initialize_db()
    return _collection


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(model="llama3.2")
    return _embedder


def add_to_db(field_name: str, summary: str, metadata: dict):
    """Embed and store a field summary with associated metadata."""
    collection = _get_collection()
    embedder = _get_embedder()

    embedding = embedder.embed_query(summary)
    doc_id = str(uuid.uuid4())

    full_metadata = {
        "field_name": field_name,
        **metadata
    }

    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[full_metadata]
    )
    print(f"[Cybertron] Stored field '{field_name}' for user '{metadata.get('user_name', 'unknown')}'")


def query_db(query_text: str, n_results: int = 5) -> list[dict]:
    """Retrieve the top-n most semantically relevant summaries."""
    collection = _get_collection()
    embedder = _get_embedder()

    query_embedding = embedder.embed_query(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count() or 1)
    )

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return output


def list_fields(user_name: str) -> list[str]:
    """List all stored field names for a given user."""
    collection = _get_collection()
    results = collection.get(where={"user_name": user_name})
    fields = list({m["field_name"] for m in results["metadatas"]}) if results["metadatas"] else []
    return fields


if __name__ == "__main__":
    initialize_db()
    print("[Cybertron] Self-test: adding a dummy record...")
    add_to_db(
        field_name="test_field",
        summary="This is a test summary about daily routines.",
        metadata={"user_name": "Test_User", "session_date": "2025-01-01"}
    )
    results = query_db("daily routines")
    print(f"[Cybertron] Query returned {len(results)} result(s).")
    fields = list_fields("Test_User")
    print(f"[Cybertron] Fields for Test_User: {fields}")
