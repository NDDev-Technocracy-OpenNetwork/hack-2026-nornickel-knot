"""Мини-сайт для теста MVP: вопрос -> граф -> ответ с источниками.

Запуск:
  uv run uvicorn src.webapp.app:app --reload --port 8000
Открыть: http://localhost:8000
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "graph"))
sys.path.insert(0, str(ROOT / "src" / "query"))

from ask import answer_question, get_client  # noqa: E402
from graph_store import load_graph  # noqa: E402

app = FastAPI(title="Норникель -- Научный клубок MVP")

_state: dict = {}


@app.on_event("startup")
def startup() -> None:
    _state["graph"] = load_graph()
    _state["client"] = get_client()
    g = _state["graph"]
    by_type: dict[str, int] = {}
    for _, data in g.nodes(data=True):
        by_type[data["type"]] = by_type.get(data["type"], 0) + 1
    _state["stats"] = {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "by_type": by_type,
    }


class Question(BaseModel):
    question: str


@app.get("/api/stats")
def stats():
    return _state["stats"]


@app.post("/api/ask")
def ask(q: Question):
    result = answer_question(_state["graph"], _state["client"], q.question)
    return result


@app.get("/")
def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
