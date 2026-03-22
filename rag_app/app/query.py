"""Semantic search over the vector store."""

from __future__ import annotations

from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

from app.llm import embed_text
from app.vector_store import InMemoryVectorStore


def semantic_search(
    query: str,
    store: InMemoryVectorStore,
    model: SentenceTransformer,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Embed the query and return the top matching resumes by cosine similarity.

    Pipeline: query string → embedding → similarity search → ranked results.
    """
    if not query or not query.strip():
        return []
    q_emb = embed_text(model, query)
    return store.search(q_emb, top_k=top_k)
