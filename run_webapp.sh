#!/usr/bin/env bash
# Запуск мини-сайта для теста MVP: http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"

if [ -x .venv/bin/python ]; then
  exec .venv/bin/python -m uvicorn src.webapp.app:app --host 127.0.0.1 --port "$PORT"
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run uvicorn src.webapp.app:app --host 127.0.0.1 --port "$PORT"
fi

exec python3 -m uvicorn src.webapp.app:app --host 127.0.0.1 --port "$PORT"
