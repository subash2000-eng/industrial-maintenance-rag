@echo off
echo ========================================
echo   Industrial Maintenance RAG System
echo ========================================
echo.

echo Starting Ollama...
start "Ollama" ollama serve
timeout /t 5 /nobreak

echo Starting FastAPI...
start "FastAPI" cmd /k "cd /d C:\Users\ADMIN\Desktop\industrial-maintenance-rag && venv\Scripts\activate && python api/main.py"
timeout /t 15 /nobreak

echo Starting Streamlit...
start "Streamlit" cmd /k "cd /d C:\Users\ADMIN\Desktop\industrial-maintenance-rag && venv\Scripts\activate && streamlit run frontend/app.py"

echo.
echo ========================================
echo   All services started!
echo   App: http://localhost:8501
echo   API: http://localhost:8000
echo ========================================
timeout /t 3 /nobreak