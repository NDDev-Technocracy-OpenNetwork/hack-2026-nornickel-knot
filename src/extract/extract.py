"""Chunk-level fact extraction over data/parsed/*.json using Cerebras
(gpt-oss-120b, OpenAI-compatible JSON mode). Writes one JSON file per document
into data/extracted/, with every entity/relation carrying provenance
(filename, category, chunk index, pages).

Resilient to partial failure: documents already extracted are skipped on
rerun, and a chunk that fails after all retries is logged and skipped rather
than crashing the whole batch.

MVP scope note: each document is capped to its first MAX_CHARS_PER_DOC
characters before chunking, to keep the number of API calls (and therefore
free-tier quota use) manageable across all sample documents in one night.
Full-document coverage is a scaling step for after the hackathon demo.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import ExtractionResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "data" / "parsed"
DEST_ROOT = ROOT / "data" / "extracted"

MODEL = "gpt-oss-120b"
MAX_CHARS_PER_DOC = 12000
CHUNK_SIZE = 3000
MAX_RETRIES = 3
MAX_TOKENS = 4096
DELAY_BETWEEN_CALLS = 13  # Cerebras free tier: 5 requests/minute account-wide

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


def extract_chunk(client: OpenAI, chunk_text: str) -> ExtractionResult:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": ONTOLOGY_PROMPT.format(chunk_text=chunk_text)}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=MAX_TOKENS,
            )
            raw = resp.choices[0].message.content
            return ExtractionResult.model_validate_json(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{MAX_RETRIES} after error: {str(e)[:200]} (sleep {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"extraction failed after {MAX_RETRIES} retries: {last_err}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["CEREBRAS_API_KEY"], base_url="https://api.cerebras.ai/v1")

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    parsed_files = sorted(SOURCE_ROOT.rglob("*.json"))

    total_entities, total_relations, total_calls, skipped_chunks = 0, 0, 0, 0

    for parsed_path in parsed_files:
        category_dir = DEST_ROOT / parsed_path.parent.name
        dest_path = category_dir / parsed_path.name
        if dest_path.exists():
            print(f"SKIP (already done): {parsed_path.name}")
            continue

        doc = json.loads(parsed_path.read_text(encoding="utf-8"))
        category = doc["category"]
        filename = doc["filename"]

        chunks = make_chunks(doc["pages"], MAX_CHARS_PER_DOC, CHUNK_SIZE)
        print(f"\n{filename} ({category}) -> {len(chunks)} chunk(s)")

        doc_entities, doc_relations = [], []
        for i, chunk in enumerate(chunks):
            try:
                result = extract_chunk(client, chunk["text"])
            except Exception as e:  # noqa: BLE001
                print(f"  chunk {i}: FAILED, skipping ({str(e)[:150]})")
                skipped_chunks += 1
                time.sleep(DELAY_BETWEEN_CALLS)
                continue
            total_calls += 1

            provenance = {
                "source_file": filename,
                "category": category,
                "chunk_index": i,
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
            }
            for e in result.entities:
                doc_entities.append({**e.model_dump(), "provenance": provenance})
            for r in result.relations:
                doc_relations.append({**r.model_dump(), "provenance": provenance})

            print(f"  chunk {i}: {len(result.entities)} entities, {len(result.relations)} relations")
            time.sleep(DELAY_BETWEEN_CALLS)

        total_entities += len(doc_entities)
        total_relations += len(doc_relations)

        category_dir.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(
            json.dumps(
                {"filename": filename, "category": category, "entities": doc_entities, "relations": doc_relations},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    print(f"\nDone. {total_calls} API calls ({skipped_chunks} chunks skipped after retries), "
          f"{total_entities} entities, {total_relations} relations across {len(parsed_files)} documents.")


if __name__ == "__main__":
    main()
