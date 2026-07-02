"""Chunk-level fact extraction over data/parsed/*.json using Gemini structured
output. Writes one JSON file per document into data/extracted/, with every
entity/relation carrying provenance (filename, category, chunk index, pages).

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
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import ExtractionResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "data" / "parsed"
DEST_ROOT = ROOT / "data" / "extracted"

MODEL = "gemini-2.5-flash-lite"
MAX_CHARS_PER_DOC = 12000
CHUNK_SIZE = 3000
MAX_RETRIES = 3
DELAY_BETWEEN_CALLS = 2.5  # be gentle with free-tier RPM across an unattended batch run

ONTOLOGY_PROMPT = """Ты извлекаешь структурированные факты из технического текста
по горно-металлургической отрасли (добыча, обогащение, гидро- и пирометаллургия)
для карты знаний R&D. Текст может быть на русском или английском.

Извлекай ТОЛЬКО то, что явно есть в тексте. Не выдумывай факты и цифры.

Типы сущностей: Material, Process, Equipment, Property, Experiment,
Publication, Expert, Facility.

Типы связей: uses_material, operates_at_condition, produces_output,
described_in, validated_by, contradicts.

Для canonical_name: нормализуй синонимы и переводы к одному виду
(например «электроэкстракция» и «electrowinning» -> «electrowinning»).

Если в куске текста нет явных фактов по этой онтологии — верни пустые списки.

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


def extract_chunk(client: genai.Client, chunk_text: str) -> ExtractionResult:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=ONTOLOGY_PROMPT.format(chunk_text=chunk_text),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractionResult,
                    temperature=0.1,
                ),
            )
            return ExtractionResult.model_validate_json(resp.text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{MAX_RETRIES} after error: {str(e)[:200]} (sleep {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"extraction failed after {MAX_RETRIES} retries: {last_err}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    parsed_files = sorted(SOURCE_ROOT.rglob("*.json"))

    total_entities, total_relations, total_calls = 0, 0, 0

    for parsed_path in parsed_files:
        doc = json.loads(parsed_path.read_text(encoding="utf-8"))
        category = doc["category"]
        filename = doc["filename"]

        chunks = make_chunks(doc["pages"], MAX_CHARS_PER_DOC, CHUNK_SIZE)
        print(f"\n{filename} ({category}) -> {len(chunks)} chunk(s)")

        doc_entities, doc_relations = [], []
        for i, chunk in enumerate(chunks):
            result = extract_chunk(client, chunk["text"])
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

        dest_dir = DEST_ROOT / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / parsed_path.name
        dest_path.write_text(
            json.dumps(
                {"filename": filename, "category": category, "entities": doc_entities, "relations": doc_relations},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    print(f"\nDone. {total_calls} API calls, {total_entities} entities, {total_relations} relations "
          f"across {len(parsed_files)} documents.")


if __name__ == "__main__":
    main()
