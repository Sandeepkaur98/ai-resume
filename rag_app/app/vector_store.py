"""
In-memory vector store with cosine similarity search (NumPy only).

No network calls: vectors and metadata live in process memory for the demo.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (L2-normalized inputs yield dot product)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class VectorRecord:
    """One indexed resume: embedding plus metadata for display and debugging."""

    id: str
    filename: str
    text: str
    embedding: np.ndarray


@dataclass
class InMemoryVectorStore:
    """Append-only in-memory store; uses cosine similarity for nearest neighbors."""

    records: List[VectorRecord] = field(default_factory=list)
    _dim: Optional[int] = None

    def clear(self) -> None:
        self.records.clear()
        self._dim = None

    def add(
        self,
        filename: str,
        text: str,
        embedding: np.ndarray,
        record_id: Optional[str] = None,
    ) -> str:
        """Add one document. Returns the assigned record id."""
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if self._dim is None:
            self._dim = int(vec.shape[0])
        elif vec.shape[0] != self._dim:
            raise ValueError(
                f"Embedding dim {vec.shape[0]} does not match store dim {self._dim}."
            )
        rid = record_id or str(uuid.uuid4())
        self.records.append(
            VectorRecord(
                id=rid,
                filename=filename,
                text=text,
                embedding=vec,
            )
        )
        return rid

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return up to ``top_k`` results sorted by cosine similarity (higher is better).

        Each result is a dict: id, filename, score, text_preview.
        """
        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        if not self.records:
            return []

        scores: List[Tuple[float, VectorRecord]] = []
        for rec in self.records:
            s = cosine_similarity(q, rec.embedding)
            scores.append((s, rec))

        scores.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for s, rec in scores[: max(1, top_k)]:
            preview = rec.text.strip().replace("\n", " ")
            if len(preview) > 400:
                preview = preview[:397] + "..."
            out.append(
                {
                    "id": rec.id,
                    "filename": rec.filename,
                    "score": float(s),
                    "text_preview": preview,
                }
            )
        return out
