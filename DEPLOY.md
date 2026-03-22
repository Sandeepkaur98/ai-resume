# Deploying the AI Resume Semantic Search app

The runnable app lives under **`rag_app/`**. Dependencies are pinned in **`requirements.txt`** at the repository root (this folder) and duplicated in **`rag_app/requirements.txt`** so hosts that only install from the root still get the same stack.

Use **Python 3.11** (see `runtime.txt`; Streamlit Cloud expects `python-3.11`). For **Heroku**, replace `runtime.txt` with a [supported release](https://devcenter.heroku.com/articles/python-runtimes) such as `python-3.11.9` if the buildpack requires a patch version.

**Pins vs. course handout:** Older handouts may list `torch==2.1.2`, `transformers==4.36.2`, and `sentence-transformers==2.2.2`. Those combinations often **fail on Python 3.12+** (no `torch` wheel) or **force a Rust build of `tokenizers`** on Windows. This repo pins **`torch==2.6.0`**, **`transformers==4.46.3`**, **`sentence-transformers==3.0.1`**, **`streamlit==1.41.0`** (needs NumPy 2), and **`numpy==2.2.2`** so `pip install -r requirements.txt` works on **Python 3.11–3.13** for both cloud and local machines.

## Streamlit Community Cloud

1. Push this repo to GitHub (this `ai-resume` folder can be the repo root, or use a monorepo and point the app at this subfolder).
2. [Share app → New app](https://share.streamlit.io/).
3. **Main file path:** `rag_app/app.py`
4. **Branch:** your default branch.
5. If the GitHub repo root is **above** this folder, set **App URL** / **Subpath** in Advanced settings to this project directory, or move `rag_app` + root `requirements.txt` to the repo root you connect.

Streamlit Cloud reads **`requirements.txt` from the repo root** — that is why the root file duplicates `rag_app/requirements.txt`.

## Heroku / Railway / Render (generic)

- **Build:** `pip install -r requirements.txt`
- **Start:** use the included **`Procfile`** (`streamlit run rag_app/app.py` with `$PORT` and `0.0.0.0`).

Set the platform’s **Python version** to **3.11.x** to align with `runtime.txt`.

## Optional FastAPI service

The optional `rag_app/api.py` is **not** included in the pinned deploy requirements. To run it locally or on a second service:

```bash
pip install fastapi "uvicorn[standard]" python-multipart
uvicorn rag_app.api:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

- **`torch` install fails:** Use Python 3.11 (not 3.13) on the host, or adjust `torch` / `numpy` pins for your platform after checking [PyTorch get-started](https://pytorch.org/get-started/locally/).
- **Wrong `requirements.txt` picked:** Ensure the service runs with working directory at the folder that contains **`requirements.txt`** and **`rag_app/`**.
