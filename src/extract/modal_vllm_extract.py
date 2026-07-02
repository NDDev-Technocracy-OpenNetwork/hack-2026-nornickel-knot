"""Batch-извлечение фактов через self-hosted Qwen на H100 (vLLM offline batch --
не сервер, один процесс грузит модель и прогоняет все чанки разом).

Решает проблему дневных лимитов бесплатных API (Cerebras 1M ток/день,
OpenRouter free-tier): свой GPU, без лимита на запросы, вся пачка чанков
за один прогон vLLM с continuous batching.

Запуск:
  modal run src/extract/modal_vllm_extract.py
"""

import json
import pathlib

import modal

app = modal.App("nornikel-vllm-extract")
# cwd, не __file__.parents -- этот модуль исполняется и локально (cwd=корень
# проекта, т.к. modal run запускается оттуда), и удалённо в контейнере Modal,
# где файл лежит на другой глубине (/root/modal_vllm_extract.py) и parents[2]
# уже не существует. Remote-сторона (run_batch) эти пути не использует вовсе.
ROOT = pathlib.Path.cwd()

SOURCE_ROOT = ROOT / "data" / "parsed"
DEST_ROOT = ROOT / "data" / "extracted"
MAX_CHARS_PER_DOC = 12000
CHUNK_SIZE = 3000

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.6.6", "transformers==4.46.3", "tokenizers==0.20.3", "pydantic")
)

# Скопировано из extract.py / schema.py напрямую -- не импортирую,
# чтобы local_entrypoint не тянул openai-клиент, который тут не нужен.

ONTOLOGY_PROMPT = """Ты извлекаешь структурированные факты из технического текста
по горно-металлургической отрасли (добыча, обогащение, гидро- и пирометаллургия)
для карты знаний R&D. Текст может быть на русском или английском.

Извлекай ТОЛЬКО то, что явно есть в тексте. Не выдумывай факты и цифры.
Не создавай отдельную сущность Publication для каждого номера ссылки из списка
литературы, если о ней в тексте больше ничего не сказано -- это мусор, не факт.

Типы сущностей: Material, Process, Equipment, Property, Experiment,
Publication, Expert, Facility.

Типы связей: uses_material, operates_at_condition, produces_output,
described_in, validated_by, contradicts.

Для canonical_name: нормализуй синонимы и переводы к одному виду
(например «электроэкстракция» и «electrowinning» -> «electrowinning»).

Если в куске текста нет явных фактов по этой онтологии -- верни пустые списки.

Верни строго JSON без пояснений, по схеме:
{{"entities": [{{"id": str, "type": str, "name": str, "canonical_name": str}}],
  "relations": [{{"source_id": str, "relation": str, "target_id": str,
                  "numeric_value": float|null, "unit": str|null,
                  "condition": str|null, "confidence": float}}]}}

Текст:
---
{chunk_text}
---
"""


def make_chunks(pages: list[dict], max_total_chars: int, chunk_size: int) -> list[dict]:
    combined = []
    total = 0
    for p in pages:
        if total >= max_total_chars:
            break
        remaining = max_total_chars - total
        text = p["text"][:remaining]
        if text.strip():
            combined.append((p["page"], text))
        total += len(text)

    chunks = []
    buf, buf_pages = "", []
    for page_num, text in combined:
        buf += text
        buf_pages.append(page_num)
        if len(buf) >= chunk_size:
            chunks.append({"text": buf, "page_start": buf_pages[0], "page_end": buf_pages[-1]})
            buf, buf_pages = "", []
    if buf.strip():
        chunks.append({"text": buf, "page_start": buf_pages[0], "page_end": buf_pages[-1]})
    return chunks


def collect_pending_chunks():
    """Те же документы/чанки что и extract.py, но только уже НЕ извлечённые."""
    pending = []
    for parsed_path in sorted(SOURCE_ROOT.rglob("*.json")):
        category_dir = DEST_ROOT / parsed_path.parent.name
        dest_path = category_dir / parsed_path.name
        if dest_path.exists():
            continue
        doc = json.loads(parsed_path.read_text(encoding="utf-8"))
        chunks = make_chunks(doc["pages"], MAX_CHARS_PER_DOC, CHUNK_SIZE)
        for i, chunk in enumerate(chunks):
            pending.append({
                "filename": doc["filename"], "category": doc["category"],
                "chunk_index": i, "page_start": chunk["page_start"], "page_end": chunk["page_end"],
                "text": chunk["text"],
            })
    return pending


@app.function(gpu="H100", image=image, timeout=3 * 3600)
def run_batch(items_json: str) -> str:
    from pydantic import BaseModel
    from typing import Literal, Optional
    from vllm import LLM, SamplingParams

    class Entity(BaseModel):
        id: str
        type: Literal["Material", "Process", "Equipment", "Property", "Experiment", "Publication", "Expert", "Facility"]
        name: str
        canonical_name: str

    class Relation(BaseModel):
        source_id: str
        relation: Literal["uses_material", "operates_at_condition", "produces_output", "described_in", "validated_by", "contradicts"]
        target_id: str
        numeric_value: Optional[float] = None
        unit: Optional[str] = None
        condition: Optional[str] = None
        confidence: float

    class ExtractionResult(BaseModel):
        entities: list[Entity]
        relations: list[Relation]

    items = json.loads(items_json)
    print(f"Загружаю {MODEL_ID} ...", flush=True)
    llm = LLM(model=MODEL_ID, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92)

    prompts = [ONTOLOGY_PROMPT.format(chunk_text=it["text"]) for it in items]
    sampling = SamplingParams(temperature=0.1, max_tokens=4096)

    print(f"Генерирую {len(prompts)} чанков одним батчем...", flush=True)
    outputs = llm.generate(prompts, sampling)

    by_doc: dict[str, dict] = {}
    ok, failed = 0, 0
    for it, out in zip(items, outputs):
        raw = out.outputs[0].text
        try:
            parsed = ExtractionResult.model_validate_json(raw)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  parse FAILED: {it['filename']} chunk {it['chunk_index']}: {str(e)[:150]}", flush=True)
            continue
        ok += 1
        key = f"{it['category']}/{it['filename']}"
        entry = by_doc.setdefault(key, {"filename": it["filename"], "category": it["category"],
                                         "entities": [], "relations": []})
        provenance = {"source_file": it["filename"], "category": it["category"],
                      "chunk_index": it["chunk_index"], "page_start": it["page_start"], "page_end": it["page_end"]}
        for e in parsed.entities:
            entry["entities"].append({**e.model_dump(), "provenance": provenance})
        for rel in parsed.relations:
            entry["relations"].append({**rel.model_dump(), "provenance": provenance})

    print(f"Готово: {ok} чанков разобрано, {failed} не распарсились, {len(by_doc)} документов.", flush=True)
    return json.dumps({"by_doc": by_doc, "ok": ok, "failed": failed}, ensure_ascii=False)


@app.local_entrypoint()
def main():
    pending = collect_pending_chunks()
    if not pending:
        print("Нечего извлекать -- всё уже готово.", flush=True)
        return
    print(f"К извлечению: {len(pending)} чанков из документов без готового extracted/*.json", flush=True)

    result_json = run_batch.remote(json.dumps(pending, ensure_ascii=False))
    result = json.loads(result_json)
    by_doc = result["by_doc"]

    for key, entry in by_doc.items():
        dest_path = DEST_ROOT / entry["category"] / entry["filename"].replace(".pdf", ".json")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {dest_path.relative_to(ROOT)}  "
              f"({len(entry['entities'])} entities, {len(entry['relations'])} relations)", flush=True)

    print(f"\nГотово. {result['ok']} чанков разобрано, {result['failed']} не распарсились. "
          f"{len(by_doc)} документов записано.", flush=True)

    print(f"\nГотово. {ok} чанков разобрано, {failed} не распарсились. "
          f"{len(by_doc)} документов записано.", flush=True)
