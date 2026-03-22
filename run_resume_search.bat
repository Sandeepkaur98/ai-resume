@echo off
cd /d "%~dp0"
echo Installing deps if needed...
pip install -q -r rag_app\requirements.txt
echo.
echo Starting AI Resume Semantic Search...
echo Open the URL shown below (usually http://localhost:8501)
echo.
streamlit run rag_app\app.py
pause
