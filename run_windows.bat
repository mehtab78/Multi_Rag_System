@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ============================================================
echo LocalRAG Windows Runner
echo ============================================================
echo.

where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
    ) else (
        echo ERROR: Python was not found. Install Python 3.10+ and add it to PATH.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Installing Python dependencies...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

python -m pip install -e .
if errorlevel 1 (
    echo ERROR: Failed to install project dependencies.
    pause
    exit /b 1
)

if not exist ".env" (
    echo.
    echo WARNING: .env file not found.
    echo Create a .env file with:
    echo GEMINI_API_KEY=your_api_key_here
    echo.
    pause
    exit /b 1
)

findstr /b /c:"GEMINI_API_KEY=" ".env" >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: GEMINI_API_KEY is missing from .env.
    echo Add this line:
    echo GEMINI_API_KEY=your_api_key_here
    echo.
    pause
    exit /b 1
)

where docker >nul 2>nul
if not errorlevel 1 (
    echo Starting OpenSearch and Ollama with Docker Compose...
    docker compose -f docker-compose.yml up -d

    echo Ensuring embedding model is available...
    docker exec ollama ollama pull nomic-embed-text
) else (
    echo.
    echo WARNING: Docker was not found on PATH.
    echo Start OpenSearch and Ollama manually before querying the app.
    echo.
)

echo.
echo Launching LocalRAG at http://localhost:7860
echo Press CTRL+C to stop the server.
echo.
python main.py

pause
