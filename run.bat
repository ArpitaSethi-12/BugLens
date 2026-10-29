@echo off
REM Start BugLens on Windows. First run installs everything into .venv
cd /d "%~dp0"

if not exist .venv (
  echo Creating virtual environment, first run only...
  python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip -q
  .venv\Scripts\pip install -r requirements.txt
)
if not exist .env copy .env.example .env

echo.
echo BugLens is starting at http://localhost:8000
echo.
start "" http://localhost:8000
.venv\Scripts\python -m uvicorn app.main:app --port 8000
