#!/usr/bin/env bash
# Start BugLens on macOS or Linux. First run installs everything into .venv
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtual environment (first run only)..."
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip -q
  ./.venv/bin/pip install -r requirements.txt
fi
[ -f .env ] || cp .env.example .env

PORT=${PORT:-8000}
echo ""
echo "BugLens is starting at http://localhost:$PORT"
echo ""
# open the browser only once the server answers (models take a few seconds to load)
( for i in $(seq 1 120); do curl -s "http://localhost:$PORT/api/status" >/dev/null 2>&1 && break; sleep 1; done
  open "http://localhost:$PORT" 2>/dev/null || xdg-open "http://localhost:$PORT" 2>/dev/null || true ) &
./.venv/bin/python -m uvicorn app.main:app --port "$PORT"
