#!/usr/bin/env bash
# Запуск мини-сайта для теста MVP: http://localhost:8000
set -e
cd "$(dirname "$0")"
uv run uvicorn src.webapp.app:app --port 8000
