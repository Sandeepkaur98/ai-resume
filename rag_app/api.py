"""
Optional FastAPI service for programmatic ingest/search (same in-memory store per process).

Run:
    uvicorn rag_app.api:app --reload
Or from rag_app directory:
    uvicorn api:app --reload

Note: Data is not persisted; restarting the process clears the index.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow ``from app.…`` when launching uvicorn from the repository root.
_RAG_APP_ROOT = Path(__file__).resolve().parent
if str(_RAG_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_APP_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.ingest import ingest_pdf_file
from app.llm import get_embedding_model
from app.query import semantic_search
from app.vector_store import InMemoryVectorStore

app = FastAPI(title="Resume Semantic Search API", version="1.0.0")

_store = InMemoryVectorStore()
_model = get_embedding_model()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class SearchResult(BaseModel):
    id: str
    filename: str
    score: float
    text_preview: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "indexed": len(_store.records)}


@app.post("/ingest", summary="Upload one PDF and add it to the index")
async def ingest(file: UploadFile = File(...)) -> Dict[str, str]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Expected a PDF file.")
    data = await file.read()
    try:
        rid = ingest_pdf_file(_store, _model, file.filename, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"id": rid, "filename": file.filename}


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    if not _store.records:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")
    raw = semantic_search(body.query, _store, _model, top_k=body.top_k)
    return SearchResponse(
        results=[SearchResult(**r) for r in raw],
    )


@app.post("/clear")
def clear() -> Dict[str, str]:
    _store.clear()
    return {"status": "cleared"}
