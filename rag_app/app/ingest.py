"""Ingest PDF resumes into the in-memory vector store."""

from __future__ import annotations

from typing import BinaryIO, Union

from sentence_transformers import SentenceTransformer

from app.llm import embed_text
from app.utils import extract_text_from_pdf
from app.vector_store import InMemoryVectorStore


def ingest_pdf_file(
    store: InMemoryVectorStore,
    model: SentenceTransformer,
    filename: str,
    file_obj: Union[BinaryIO, bytes],
) -> str:
    """
    Extract text from a PDF, embed it, and append to ``store``.

    Returns the new record id.
    """
    # Streamlit UploadedFile may have been read already in the same request.
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    text = extract_text_from_pdf(file_obj)
    if not text:
        raise ValueError(f"No extractable text in PDF: {filename}")
    emb = embed_text(model, text)
    return store.add(filename=filename, text=text, embedding=emb)
