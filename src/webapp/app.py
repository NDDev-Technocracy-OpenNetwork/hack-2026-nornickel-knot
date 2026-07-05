"""Мини-сайт для теста MVP: вопрос -> граф -> ответ с источниками.

Запуск:
  uv run uvicorn src.webapp.app:app --reload --port 8000
Открыть: http://localhost:8000
"""

import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "graph"))
sys.path.insert(0, str(ROOT / "src" / "query"))
sys.path.insert(0, str(Path(__file__).parent))  # для import ingest

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


def _recount_stats() -> None:
    g = _state["graph"]
    by_type: dict[str, int] = {}
    for _, data in g.nodes(data=True):
        by_type[data["type"]] = by_type.get(data["type"], 0) + 1
    _state["stats"] = {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                       "by_type": by_type}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Живая загрузка PDF: извлекаем факты, вливаем в граф без перезапуска.
    Синхронно (~30-90 сек на документ) -- для демо/MVP одного пользователя
    этого достаточно, очередь и прогресс-бар -- продовая доработка."""
    import ingest
    if not (file.filename or "").lower().endswith(".pdf"):
        return {"ok": False, "error": "Пока поддерживается только PDF."}
    data = await file.read()
    result = ingest.ingest_document(_state["graph"], data, file.filename)
    if result.get("ok"):
        _recount_stats()
    return result


def _graph_payload(g, keys, seed_keys=frozenset(), max_edges: int = 2500) -> dict:
    """Общий формат под canvas-рендерер: узлы + рёбра между ними."""
    keep = set(keys)
    sub = g.subgraph(keep)
    degrees = dict(sub.degree())
    nodes = [
        {"id": k, "label": g.nodes[k]["canonical_name"], "type": g.nodes[k]["type"],
         "degree": degrees.get(k, 0), "mentions": g.nodes[k].get("mentions", 1),
         "full_degree": g.degree(k),  # полная степень в БОЛЬШОМ графе, не в срезе
         "seed": k in seed_keys}
        for k in keep
    ]
    edges, seen = [], set()
    for u, v, data in sub.edges(data=True):
        if u == v:
            continue
        pair = (u, v) if u < v else (v, u)
        if pair in seen:
            continue
        seen.add(pair)
        edges.append({"source": u, "target": v, "relation": data["relation"]})
        if len(edges) >= max_edges:
            break
    return {"nodes": nodes, "edges": edges}


@app.get("/api/graph/node")
def graph_node(key: str):
    """Карточка узла для панели: источники, соседи с типами связей.
    Делает граф исследуемым, а не просто картинкой."""
    g = _state["graph"]
    if not g.has_node(key):
        return {"ok": False, "error": "узел не найден"}
    data = g.nodes[key]
    # соседи с фактами -- собираем в обе стороны, сгруппировано по связи
    facts = []
    for u, v, ed in list(g.out_edges(key, data=True))[:120]:
        facts.append({"dir": "out", "relation": ed["relation"],
                      "other": g.nodes[v]["canonical_name"], "other_key": v,
                      "value": ed.get("numeric_value"), "unit": ed.get("unit"),
                      "condition": ed.get("condition"), "source": ed.get("source_file"),
                      "pages": [ed.get("page_start"), ed.get("page_end")]})
    for u, v, ed in list(g.in_edges(key, data=True))[:120]:
        facts.append({"dir": "in", "relation": ed["relation"],
                      "other": g.nodes[u]["canonical_name"], "other_key": u,
                      "value": ed.get("numeric_value"), "unit": ed.get("unit"),
                      "condition": ed.get("condition"), "source": ed.get("source_file"),
                      "pages": [ed.get("page_start"), ed.get("page_end")]})
    return {
        "ok": True, "key": key,
        "label": data["canonical_name"], "type": data["type"],
        "mentions": data.get("mentions", 1),
        "surface_forms": sorted(data.get("surface_forms", []))[:8],
        "sources": sorted(data.get("sources", []))[:10],
        "n_sources": len(data.get("sources", [])),
        "degree": g.degree(key),
        "facts": facts[:60],
    }


@app.get("/api/graph/expand")
def graph_expand(key: str, limit: int = 30):
    """Раскрытие соседей узла -- Neo4j-подобное исследование графа кликом."""
    import ask as ask_mod
    g = _state["graph"]
    if not g.has_node(key):
        return {"nodes": [], "edges": []}
    sub = ask_mod.neighborhood(g, [key], hops=1, max_nodes=min(max(limit, 5), 60))
    return _graph_payload(g, list(sub.nodes), seed_keys={key})


@app.get("/api/graph/browse")
def graph_browse(q: str = "", types: str = "", limit: int = 250):
    """Obsidian-режим: топ узлов по упоминаемости, фильтр по типу и подстроке.
    limit ограничен: наивная O(n^2) силовая укладка в браузере комфортна
    до ~300 узлов."""
    g = _state["graph"]
    limit = min(max(limit, 10), 400)
    type_set = {t for t in types.split(",") if t} if types else None
    ql = q.strip().lower()

    scored = []
    for key, data in g.nodes(data=True):
        if type_set and data["type"] not in type_set:
            continue
        if ql and ql not in data["canonical_name"].lower() \
              and not any(ql in s.lower() for s in data["surface_forms"]):
            continue
        scored.append((data.get("mentions", 1), key))
    scored.sort(reverse=True)
    keys = [k for _, k in scored[:limit]]
    payload = _graph_payload(g, keys)
    payload["total_matched"] = len(scored)
    return payload


class DiscoverQuery(BaseModel):
    query: str


@app.post("/api/graph/discover")
def graph_discover(dq: DiscoverQuery):
    """Нейро-режим: Harmonic-эмбеддинги + сущностный поиск находят смысловые
    затравки, граф раскрывается вокруг них. Никакой LLM -- ~2 сек."""
    import ask as ask_mod
    g = _state["graph"]
    seeds, _scored = ask_mod.retrieve_seed_nodes_harmonic(g, dq.query)
    if not seeds:
        return {"nodes": [], "edges": [], "total_matched": 0}
    seed_keys = seeds[:40]
    sub = ask_mod.neighborhood(g, seed_keys, hops=1, max_nodes=220)
    payload = _graph_payload(g, list(sub.nodes), seed_keys=set(seed_keys))
    payload["total_matched"] = len(seeds)
    return payload


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "nornikel.html"))


@app.get("/harmonic")
def harmonic_page():
    return FileResponse(str(STATIC_DIR / "harmonic.html"))


@app.get("/nornikel")
def nornikel_page():
    return FileResponse(str(STATIC_DIR / "nornikel.html"))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
