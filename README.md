# Repository: Endee fork + AI Resume Semantic Search

This repository contains:

- **`ai-resume/`** — Fork of the **Endee** project (C++ sources, `src/`, `infra/`, `tests/`, `third_party/`, etc.). **Do not remove or modify** these trees for the Python assignment unless you intend to work on Endee itself.
- **`rag_app/`** — **AI Resume Semantic Search** application (Python): PDF upload, local embeddings, in-memory cosine-similarity search, Streamlit UI.

## Quick start (semantic search app)

```bash
pip install -r rag_app/requirements.txt
streamlit run rag_app/app.py
```

**Windows:** run **`run_resume_search.bat`** in this folder to install dependencies (if needed) and launch the app in your browser.

Full documentation: **[rag_app/README.md](rag_app/README.md)**.
