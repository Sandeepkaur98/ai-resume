"""
Embedding model wrapper (sentence-transformers).

This module is named ``llm`` to leave room for a future generative LLM in the
RAG stack (retrieve → optionally augment with an LLM answer). For this
assignment, retrieval uses dense embeddings only.
"""

from __future__ import annotations

import logging
import threading
from typing import List

# Transformers 5.x logs a long "LOAD REPORT" for UNEXPECTED keys (e.g.
# `embeddings.position_ids` on BERT/MiniLM). Those keys are optional buffers
# in the checkpoint and do not change inference; suppress the noisy warning.
logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)

import numpy as np
from sentence_transformers import SentenceTransformer

# Default model: small, fast, good for semantic similarity on short documents.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_loaded_model_name: str | None = None
_model_lock = threading.Lock()


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load (once per model name) and return the shared SentenceTransformer instance."""
    global _model, _loaded_model_name
    with _model_lock:
        if _model is None or _loaded_model_name != model_name:
            _model = SentenceTransformer(model_name)
            _loaded_model_name = model_name
        return _model


def embed_text(model: SentenceTransformer, text: str) -> np.ndarray:
    """Encode a single string to a 1-D float32 embedding vector."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")
    vec = model.encode(text.strip(), convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Encode multiple strings; returns shape (n, dim)."""
    cleaned = [t.strip() if t else "" for t in texts]
    if not any(cleaned):
        raise ValueError("No non-empty texts to embed.")
    arr = model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(arr, dtype=np.float32)
