"""Живая загрузка пользовательского документа в граф знаний.

Пользователь загружает PDF на сайте -> парсим постранично -> чанки ->
извлечение фактов той же онтологией через Cerebras (тот же промпт, что и у
основного пайплайна) -> вливаем сущности/связи в граф В ПАМЯТИ работающего
приложения (без перезапуска) -> пересчитываем Harmonic-эмбеддинг документа,
чтобы новый документ сразу находился и семантическим ретривом тоже.

Всё персистится: parsed/extracted JSON на диск (переживает перезапуск,
попадает в будущие пересборки графа), граф -> pkl, эмбеддинги -> .pt.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "extract"))
sys.path.insert(0, str(ROOT / "src" / "graph"))
sys.path.insert(0, str(ROOT / "src" / "harmonic_train"))
sys.path.insert(0, str(ROOT / "src" / "query"))

import torch  # noqa: E402
from openai import OpenAI  # noqa: E402

from extract import ONTOLOGY_PROMPT, make_chunks  # noqa: E402
from graph_store import node_key, save_graph  # noqa: E402
from harmonic_embed import embed_text, load_model  # noqa: E402
from schema import ExtractionResult  # noqa: E402

UPLOAD_CATEGORY = "Загруженные пользователем"
# один документ на живой загрузке: ограничиваем размер, чтобы демо-загрузка
# занимала ~30-90 сек, а не 10 минут. 60000 символов ~= 20 чанков.
MAX_UPLOAD_CHARS = 60000
CHUNK_SIZE = 3000
PARALLEL_CALLS = 4  # бережём rate-limit Cerebras free-tier

PARSED_DIR = ROOT / "data" / "parsed" / UPLOAD_CATEGORY
EXTRACTED_DIR = ROOT / "data" / "extracted" / UPLOAD_CATEGORY
DOC_EMBEDDINGS_PATH = ROOT / "data" / "harmonic_runs" / "doc_embeddings.pt"


def parse_pdf_bytes(data: bytes) -> list[dict]:
    import fitz
    import pymupdf4llm

    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i in range(doc.page_count):
        try:
            text = pymupdf4llm.to_markdown(doc, pages=[i], show_progress=False)
        except Exception:  # noqa: BLE001 -- битые страницы не роняют документ
            text = doc[i].get_text("text")
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def _extract_client() -> tuple[OpenAI, str]:
    """Yandex -- условие хакатона: только API с доступом из РФ."""
    import os
    model = f"gpt://{os.environ.get('YANDEX_FOLDER_ID', '')}/yandexgpt/latest"
    return OpenAI(api_key=os.environ["YANDEX_API_KEY"],
                  base_url="https://llm.api.cloud.yandex.net/v1"), model


def _clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text.strip()


def _extract_chunk_yandex(client: OpenAI, model: str, chunk_text: str) -> ExtractionResult:
    """Как extract.extract_chunk, но без response_format=json_object --
    OpenAI-совместимый эндпоинт Яндекса его не гарантирует, парсим сами."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": ONTOLOGY_PROMPT.format(chunk_text=chunk_text)}],
        temperature=0.1,
        max_tokens=4000,
    )
    raw = _clean_json(resp.choices[0].message.content or "")
    return ExtractionResult.model_validate_json(raw)


def merge_into_graph(g, entry: dict) -> tuple[int, int]:
    """Та же логика, что build_graph() в graph_store.py, но для одного
    документа поверх живого графа. Возвращает (новых узлов, новых рёбер)."""
    n_nodes_before = g.number_of_nodes()
    n_edges_before = g.number_of_edges()

    local_to_key: dict[str, str] = {}
    for e in entry["entities"]:
        key = node_key(e["type"], e["canonical_name"])
        local_to_key[e["id"]] = key
        if g.has_node(key):
            g.nodes[key]["surface_forms"].add(e["name"])
            g.nodes[key]["mentions"] += 1
            g.nodes[key]["sources"].add(e["provenance"]["source_file"])
        else:
            g.add_node(
                key,
                type=e["type"],
                canonical_name=e["canonical_name"],
                surface_forms={e["name"]},
                mentions=1,
                sources={e["provenance"]["source_file"]},
            )

    for r in entry["relations"]:
        src = local_to_key.get(r["source_id"])
        tgt = local_to_key.get(r["target_id"])
        if src is None or tgt is None or not g.has_node(src) or not g.has_node(tgt):
            continue
        # тот же фильтр, что в build_graph: affiliated_with только от Expert
        if r["relation"] == "affiliated_with" and g.nodes[src].get("type") != "Expert":
            continue
        g.add_edge(
            src, tgt,
            relation=r["relation"],
            numeric_value=r["numeric_value"],
            unit=r["unit"],
            condition=r["condition"],
            confidence=r["confidence"],
            source_file=r["provenance"]["source_file"],
            category=r["provenance"]["category"],
            page_start=r["provenance"]["page_start"],
            page_end=r["provenance"]["page_end"],
        )

    return g.number_of_nodes() - n_nodes_before, g.number_of_edges() - n_edges_before


def update_doc_embedding(filename: str, fulltext: str) -> None:
    """Пересчитывает Harmonic-эмбеддинг нового документа и обновляет и файл
    на диске, и кэш в памяти ask-модуля (чтобы работало без перезапуска)."""
    model = load_model()
    vec = embed_text(fulltext, model=model)
    entries = torch.load(DOC_EMBEDDINGS_PATH, weights_only=False)
    entries = [e for e in entries if e["filename"] != filename]
    entries.append({"filename": filename, "category": UPLOAD_CATEGORY, "vector": vec})
    torch.save(entries, DOC_EMBEDDINGS_PATH)

    import ask
    if ask._doc_embeddings is not None:
        ask._doc_embeddings = entries


def ingest_document(g, data: bytes, filename: str) -> dict:
    """Полный цикл: PDF-байты -> граф. Возвращает сводку для UI."""
    pages = parse_pdf_bytes(data)
    if not pages:
        return {"ok": False, "error": "Не удалось извлечь текст из PDF (пустой или сканированный без текстового слоя)."}

    # персистим parsed -- тем же форматом, что основной пайплайн
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    parsed_doc = {"filename": filename, "category": UPLOAD_CATEGORY, "pages": pages}
    (PARSED_DIR / f"{stem}.json").write_text(
        json.dumps(parsed_doc, ensure_ascii=False), encoding="utf-8")

    chunks = make_chunks(pages, MAX_UPLOAD_CHARS, CHUNK_SIZE)
    client, model = _extract_client()

    def run_one(chunk):
        try:
            parsed = _extract_chunk_yandex(client, model, chunk["text"])
            return chunk, parsed
        except Exception:  # noqa: BLE001
            return chunk, None

    entities, relations = [], []
    n_failed = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_CALLS) as pool:
        for chunk, parsed in pool.map(run_one, chunks):
            if parsed is None:
                n_failed += 1
                continue
            provenance = {"source_file": filename, "category": UPLOAD_CATEGORY,
                          "chunk_index": chunks.index(chunk),
                          "page_start": chunk["page_start"], "page_end": chunk["page_end"]}
            for e in parsed.entities:
                entities.append({**e.model_dump(), "provenance": provenance})
            for r in parsed.relations:
                relations.append({**r.model_dump(), "provenance": provenance})

    entry = {"filename": filename, "category": UPLOAD_CATEGORY,
             "entities": entities, "relations": relations}

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    (EXTRACTED_DIR / f"{stem}.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    new_nodes, new_edges = merge_into_graph(g, entry)
    save_graph(g)

    fulltext = "\n\n".join(p["text"] for p in pages)
    update_doc_embedding(filename, fulltext)

    return {
        "ok": True,
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "chunks_failed": n_failed,
        "entities": len(entities),
        "relations": len(relations),
        "new_graph_nodes": new_nodes,
        "new_graph_edges": new_edges,
    }
