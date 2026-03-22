"""
Streamlit UI: upload PDF resumes and run semantic search (cosine similarity).

Run from repository root:
    streamlit run rag_app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the ``app`` package resolves when Streamlit sets a different cwd.
_RAG_ROOT = Path(__file__).resolve().parent
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

import streamlit as st

from app.ingest import ingest_pdf_file
from app.llm import get_embedding_model
from app.query import semantic_search
from app.vector_store import InMemoryVectorStore

PAGE_TITLE = "AI Resume Semantic Search"
st.set_page_config(page_title=PAGE_TITLE, page_icon="📄", layout="wide")

# Light styling: tighter spacing, clearer sections (Streamlit 1.33+)
st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    .stForm { border: 1px solid rgba(49, 51, 63, 0.12); border-radius: 0.5rem; padding: 1rem 1rem 0.5rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_session() -> None:
    if "store" not in st.session_state:
        st.session_state.store = InMemoryVectorStore()
    if "model" not in st.session_state:
        with st.spinner("Loading embedding model (first run may download weights)…"):
            st.session_state.model = get_embedding_model()
    # Keys of (filename, size) already added to the vector store — avoids re-ingesting on every rerun.
    if "indexed_file_keys" not in st.session_state:
        st.session_state.indexed_file_keys = set()
    # Last search results for stable display after other interactions.
    if "last_search" not in st.session_state:
        st.session_state.last_search = None  # tuple: (query, top_k, results list)


def _file_key(f) -> tuple[str, int]:
    return (f.name, int(f.size))


def main() -> None:
    _init_session()
    store: InMemoryVectorStore = st.session_state.store
    model = st.session_state.model
    indexed_keys: set = st.session_state.indexed_file_keys

    st.title(PAGE_TITLE)
    st.caption(
        "Semantic search over uploaded resumes using sentence embeddings and cosine similarity "
        "(local in-memory index, no external vector server)."
    )

    n_docs = len(store.records)

    with st.sidebar:
        st.header("Index")
        st.metric("Resumes indexed", n_docs)
        if st.button("Clear all resumes", type="secondary", use_container_width=True):
            store.clear()
            st.session_state.indexed_file_keys = set()
            st.session_state.last_search = None
            st.success("Index cleared.")
            st.rerun()

        st.divider()
        with st.expander("How it works"):
            st.markdown(
                """
                1. **Add PDFs** — select files, then click **Add to index** (files are not re-added on every click).
                2. **Search** — enter a query and set **Top K** (how many matches you want ranked).
                3. **Results** — you can only see up to **N** matches if you only have **N** resumes indexed.
                """
            )

    col_up, col_search = st.columns((1, 1), gap="large")

    with col_up:
        st.subheader("1. Upload resumes (PDF)")
        uploaded = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Up to 200MB per file. Text is extracted locally on your machine.",
            label_visibility="visible",
        )

        b_add = st.button("Add to index", type="primary", use_container_width=True)

        if b_add:
            if not uploaded:
                st.warning("Select one or more PDF files first.")
            else:
                new_count = 0
                skipped = 0
                errors: list[str] = []
                progress = st.progress(0.0, text="Indexing…")
                for i, f in enumerate(uploaded):
                    key = _file_key(f)
                    if key in indexed_keys:
                        skipped += 1
                        progress.progress((i + 1) / len(uploaded), text="Indexing…")
                        continue
                    try:
                        if hasattr(f, "seek"):
                            f.seek(0)
                        ingest_pdf_file(store, model, f.name, f)
                        indexed_keys.add(key)
                        new_count += 1
                    except Exception as e:
                        errors.append(f"{f.name}: {e}")
                    progress.progress((i + 1) / len(uploaded), text="Indexing…")
                st.session_state.indexed_file_keys = indexed_keys
                progress.empty()
                if new_count:
                    st.success(f"Added {new_count} new resume(s) to the index.")
                if skipped:
                    st.info(f"Skipped {skipped} file(s) already in the index (same name and size).")
                for err in errors:
                    st.error(err)
                if new_count or errors:
                    st.session_state.last_search = None
                    st.rerun()

    with col_search:
        st.subheader("2. Search")
        st.caption(
            "Top K asks for up to that many ranked results. "
            f"You currently have **{n_docs}** resume(s) indexed, so you will see at most **{n_docs}** rows."
        )

        # Form keeps query + Top K together so changing Top K does not rerun the whole page
        # or re-trigger indexing; values update only when you click Search.
        with st.form("search_form", clear_on_submit=False):
            q = st.text_input(
                "Natural language query",
                placeholder='e.g. "Python backend engineer with AWS"',
                key="nl_query",
            )
            # Number input (not a slider): full range 1–50 independent of how many PDFs are indexed.
            top_k = st.number_input(
                "Top K results",
                min_value=1,
                max_value=50,
                value=5,
                step=1,
                help="Return up to this many best matches (you cannot see more rows than resumes indexed).",
                key="top_k_input",
            )
            submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

        if submitted:
            query = (q or "").strip()
            if not query:
                st.info("Enter a search query.")
                st.session_state.last_search = None
            elif n_docs == 0:
                st.warning("Add at least one PDF to the index before searching.")
                st.session_state.last_search = None
            else:
                try:
                    with st.spinner("Embedding query and ranking resumes…"):
                        results = semantic_search(query, store, model, top_k=int(top_k))
                    st.session_state.last_search = (query, int(top_k), results)
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.session_state.last_search = None

    # Results below both columns
    last = st.session_state.last_search
    if last is not None:
        query_used, top_k_used, results = last
        st.divider()
        st.subheader("Results")
        st.caption(f"Query: “{query_used}” · Top K = **{top_k_used}** · Showing **{len(results)}** match(es)")
        if not results:
            st.write("No matches.")
        else:
            for rank, r in enumerate(results, start=1):
                with st.container():
                    st.markdown(
                        f"**{rank}.** `{r['filename']}` &nbsp;·&nbsp; "
                        f"similarity **{r['score']:.4f}**"
                    )
                    st.caption(r["text_preview"])
                    st.divider()


if __name__ == "__main__":
    main()
