"""Simple ChromaDB wrapper.

This file is intentionally the only place that knows about the vector database.
If you swap ChromaDB for another store later, you only need to update this file.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import CHROMA_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL, ensure_directories


_STORE: Optional[Chroma] = None


def init_store() -> Chroma:
    """Create or return the shared Chroma store."""
    global _STORE

    if _STORE is None:
        ensure_directories()
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        _STORE = Chroma(
            collection_name="ethno_ai_memory",
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
    return _STORE


def add_document(session_id: str, category: str, text: str) -> str:
    """Add one text summary to the vector store."""
    store = init_store()
    document_id = str(uuid.uuid4())
    document = Document(
        page_content=text,
        metadata={"session_id": session_id, "category": category},
    )
    store.add_documents(documents=[document], ids=[document_id])
    return document_id


def search(session_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search for relevant documents inside one session only."""
    store = init_store()
    results = store.similarity_search(query=query, k=top_k, filter={"session_id": session_id})

    cleaned_results: List[Dict[str, Any]] = []
    for item in results:
        cleaned_results.append(
            {
                "text": item.page_content,
                "category": item.metadata.get("category", "unknown"),
                "session_id": item.metadata.get("session_id", "unknown"),
            }
        )
    return cleaned_results


def delete_session(session_id: str) -> None:
    """Delete all documents linked to one session."""
    store = init_store()
    data = store.get(where={"session_id": session_id})
    ids = data.get("ids", [])
    if ids:
        store.delete(ids=ids)
