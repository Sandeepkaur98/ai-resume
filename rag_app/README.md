# AI Resume Semantic Search (`rag_app`)

## Project overview

`rag_app` is a **local, self-contained** semantic search demo for recruiter-style resume screening. Users upload PDF resumes, the system extracts text, encodes each document with a sentence embedding model, and ranks candidates against a natural-language query using **cosine similarity** in an **in-memory vector store** (NumPy). No Docker and **no external vector database or HTTP dependency** on a separate search server.

This code lives in the **`rag_app/`** folder alongside the forked **Endee** C++ repository under `ai-resume/`. The Endee tree is **not modified** by this assignment; `rag_app` is the Python application layer.

---

## Problem statement

Recruiters often review large resume pools manually. Keyword search misses relevant candidates when wording differs from the job description, and it can return noisy matches. The goal is to retrieve resumes by **meaning** (skills, roles, experience) rather than exact string overlap.

---

## Solution

1. **Ingest**: Extract plain text from each PDF (`pypdf`).
2. **Embed**: Map resume text and the user query to dense vectors with **sentence-transformers** (`all-MiniLM-L6-v2` by default).
3. **Index**: Store vectors and metadata in an **in-memory** structure (`InMemoryVectorStore`).
4. **Search**: Compute **cosine similarity** between the query embedding and each stored embedding; return the top-K filenames with scores and short text previews.

This is a **retrieval** pipeline. A generative LLM is not required for ranking; `app/llm.py` centralizes the embedding model so a future answer-generation step could be added without changing the store API.

---

## System architecture (text diagram)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  PDF uploads    │────▶│  Text extraction │────▶│  SentenceTransformer │
│  (Streamlit/API)│     │  (pypdf)         │     │  (embeddings)        │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                             │
                                                             ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  User query     │────▶│  Same embedder   │────▶│  In-memory vectors   │
│  (natural lang) │     │                  │     │  + NumPy cosine sim   │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                             │
                                                             ▼
                                                  ┌─────────────────────┐
                                                  │  Top-K results       │
                                                  │  (score + preview)   │
                                                  └─────────────────────┘
```

---

## How Endee is intended to be used

**Endee** (in `ai-resume/`) is a high-performance **embedded search / indexing** engine aimed at C++ integrations. In a production setting, embeddings produced by this Python stack could be **indexed and queried** through Endee’s APIs for persistence, scale, and low-latency retrieval—once an Endee server or embedded library is available in your deployment.

This submission uses a **pure Python in-memory store** instead, so evaluators can run the project **without** building Endee, without Docker, and **without** `localhost:8080` or any Endee HTTP dependency. The design keeps retrieval logic modular (`vector_store.py`, `query.py`) so swapping in an Endee-backed client later would be a targeted change.

---

## Local vector database note

Because of portability and grading constraints, **vectors are held only in RAM** inside the running process. Restarting the app clears the index. This is intentional for a demo; persistence (SQLite, file-backed FAISS, or a managed vector DB) is listed under future improvements.

---

## Setup

**Python 3.10+** recommended.

From the repository root (the folder that contains `rag_app/`):

```bash
pip install -r requirements.txt
```

(`pip install -r rag_app/requirements.txt` is equivalent — same pins.)

The first run downloads the `sentence-transformers` model (network required once).

**Deploy (Streamlit Cloud, Render, etc.):** see **`../DEPLOY.md`** in the repository root. That folder also has a **`requirements.txt`** copy for hosts that only install dependencies from the repo root.

---

## Example usage

### Streamlit UI (primary)

```bash
streamlit run rag_app/app.py
```

On Windows you can double‑click **`run_resume_search.bat`** in the repository root (installs dependencies, then starts Streamlit).

1. Select one or more PDF resumes and click **Add to index** (files are not re-ingested on every page refresh; duplicates with the same name and size are skipped).
2. Enter a query (e.g. “Senior backend engineer, Python, AWS”).
3. Set **Top K** (1–50) to how many ranked matches you want, then click **Search**.

**Note:** If you only have *N* resumes indexed, you will see at most *N* rows—raising Top K above *N* does not create more rows, it only sets the cap.

4. Review similarity scores and text previews in **Results**.

### CLI (optional)

Index PDFs (ephemeral in-memory demo for the command):

```bash
python rag_app/main.py ingest resume1.pdf resume2.pdf
```

Load PDFs and run one query:

```bash
python rag_app/main.py search "machine learning engineer" --files resume1.pdf resume2.pdf --top-k 5
```

### FastAPI (optional)

```bash
uvicorn rag_app.api:app --reload
```

Use `POST /ingest` with a PDF and `POST /search` with JSON `{"query": "...", "top_k": 5}`. See `/docs` for OpenAPI.

---

## Project layout

```
rag_app/
  app/
    ingest.py       # PDF → text → embedding → store
    query.py        # Query embedding + similarity search
    vector_store.py # In-memory store + cosine similarity
    llm.py          # Embedding model wrapper
    utils.py        # PDF text extraction
  app.py            # Streamlit UI
  main.py           # Optional CLI
  api.py            # Optional FastAPI
  requirements.txt
  README.md
```

---

## Future improvements

- Persist embeddings and metadata (SQLite or file store); optional FAISS/Annoy for large corpora.
- Chunk long resumes and aggregate scores per candidate.
- Integrate **Endee** as the backing retrieval engine when a build/runtime is available.
- Optional generative step: summarize why each candidate matches the query (RAG-style).
- Auth, multi-tenant indexes, and cloud deployment.

---

## Author note

This README accompanies the refactored **AI Resume Semantic Search** assignment code under `rag_app/`.
