"""Собирает chunks.jsonl по формату, который запросил Данила (Neo4j).

Два источника чанков:
  1. Наши 106 документов -- пересобираем через pymupdf4llm с page_chunks=True,
     получаем точные page_start/page_end + markdown-структуру (заголовки,
     таблицы) + можем привязать уже извлечённые факты (mentioned_*) по
     пересечению страниц.
  2. Остальные ~1054 документа -- берём готовый плоский markdown Айтала
     (src/backend/data/preprocessed в ScienceKnot). Там НЕТ разметки по
     страницам вообще (проверено -- pymupdf4llm.to_markdown() без
     page_chunks не сохраняет номера страниц), поэтому page_start/page_end
     = null, честно, не выдумываем. Секции/таблицы всё равно определяем по
     markdown-структуре.

Facts (mentioned_*) привязываются только там, где они у нас есть (106
документов) -- для остальных пусто, потому что извлечение по ним не
запускалось (отдельная, дорогая по времени операция).
"""

import json
import re
import sys
from pathlib import Path

import pymupdf4llm

ROOT = Path(__file__).resolve().parents[2]
PDF_ROOT = ROOT / "yandex" / "Задача 2. Научный клубок"
EXTRACTED_ROOT = ROOT / "data" / "extracted"
AITAL_MD_ROOT = Path(
    "/tmp/claude-1000/-home-suicore-hacathons-nornikel/aa1dd9a2-cd19-445e-a2c9-b14c060c5492/scratchpad"
    "/ScienceKnot/src/backend/data/preprocessed"
)
OUT_PATH = ROOT / "data" / "chunks.jsonl"

HEADER_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
CHUNK_CHAR_TARGET = 2500


def classify_chunk(text: str) -> str:
    if len(TABLE_ROW_RE.findall(text)) >= 2:
        return "table"
    if "Start of picture text" in text or "figure" in text.lower():
        return "figure_caption"
    low = text.lower()
    if "вывод" in low or "заключен" in low:
        return "conclusion"
    if "эксперимент" in low or "результат" in low:
        return "experiment"
    return "text"


def current_section(text_before: str) -> str | None:
    matches = HEADER_RE.findall(text_before)
    return matches[-1].strip("* ").strip() if matches else None


def split_into_chunks(full_text: str, target: int = CHUNK_CHAR_TARGET) -> list[tuple[str, str | None]]:
    """Режет по параграфам, копит до ~target символов, помнит последний
    встреченный заголовок как section_title для каждого куска."""
    paragraphs = [p for p in full_text.split("\n\n") if p.strip()]
    chunks: list[tuple[str, str | None]] = []
    buf: list[str] = []
    buf_len = 0
    consumed_so_far = ""
    for p in paragraphs:
        buf.append(p)
        buf_len += len(p)
        consumed_so_far += p + "\n\n"
        if buf_len >= target:
            text = "\n\n".join(buf)
            chunks.append((text, current_section(consumed_so_far)))
            buf, buf_len = [], 0
    if buf:
        text = "\n\n".join(buf)
        chunks.append((text, current_section(consumed_so_far)))
    return chunks


def load_facts_by_page(document_stem: str) -> dict:
    path = None
    for p in EXTRACTED_ROOT.rglob(f"{document_stem}.json"):
        path = p
        break
    if path is None:
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in doc["entities"]}
    type_field = {
        "Material": "mentioned_materials", "Process": "mentioned_methods",
        "Property": "mentioned_properties", "Experiment": "mentioned_experiments",
        "Facility": "mentioned_organizations", "Expert": "mentioned_authors",
    }
    per_page: dict[int, dict] = {}
    for e in doc["entities"]:
        field = type_field.get(e["type"])
        if field is None:
            continue
        pg = e["provenance"]["page_start"]
        per_page.setdefault(pg, {}).setdefault(field, set()).add(e["canonical_name"])
    return per_page


def facts_for_range(per_page: dict, page_start: int, page_end: int) -> dict:
    out: dict[str, set] = {}
    for pg, fields in per_page.items():
        if page_start <= pg <= page_end:
            for field, names in fields.items():
                out.setdefault(field, set()).update(names)
    return {k: sorted(v) for k, v in out.items()}


def process_our_106() -> list[dict]:
    rows = []
    n = 0
    for pdf_path in sorted(PDF_ROOT.rglob("*.pdf")):
        stem = pdf_path.stem
        if not any(EXTRACTED_ROOT.rglob(f"{stem}.json")):
            continue
        n += 1
        try:
            pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP (convert fail) {stem}: {e}", file=sys.stderr)
            continue

        per_page_facts = load_facts_by_page(stem)
        idx = 0
        for page in pages:
            page_no = page["metadata"].get("page_number", 1)
            text = page["text"].strip()
            if not text:
                continue
            for chunk_text, section in split_into_chunks(text):
                cid = f"{stem}::chunk{idx}"
                rows.append({
                    "document_id": stem, "source_filename": pdf_path.name,
                    "markdown_filename": f"{stem}.md",
                    "chunk_id": cid, "chunk_index": idx,
                    "chunk_type": classify_chunk(chunk_text),
                    "section_title": section, "page_start": page_no, "page_end": page_no,
                    "text": chunk_text, "token_count": len(chunk_text) // 4,
                    "char_count": len(chunk_text), "language": "ru",
                    "extraction_quality": "ok",
                    "metadata": facts_for_range(per_page_facts, page_no, page_no),
                })
                idx += 1
        print(f"  [{n}/106] {stem}: {idx} chunks")
    return rows


def process_rest_from_aital() -> list[dict]:
    if not AITAL_MD_ROOT.exists():
        print(f"Айтал md не найден по {AITAL_MD_ROOT} -- пропускаю этот источник", file=sys.stderr)
        return []
    our_stems = {p.stem for p in PDF_ROOT.rglob("*.pdf") if any(EXTRACTED_ROOT.rglob(f"{p.stem}.json"))}
    rows = []
    for md_path in sorted(AITAL_MD_ROOT.rglob("*.md")):
        stem = md_path.stem
        if stem in our_stems:
            continue  # уже обработан точным путём выше
        text = md_path.read_text(encoding="utf-8", errors="replace")
        idx = 0
        for chunk_text, section in split_into_chunks(text):
            cid = f"{stem}::chunk{idx}"
            rows.append({
                "document_id": stem, "source_filename": f"{stem}.pdf",
                "markdown_filename": md_path.name,
                "chunk_id": cid, "chunk_index": idx,
                "chunk_type": classify_chunk(chunk_text),
                "section_title": section, "page_start": None, "page_end": None,
                "text": chunk_text, "token_count": len(chunk_text) // 4,
                "char_count": len(chunk_text), "language": "ru",
                "extraction_quality": "no_page_data",
                "metadata": {},
            })
            idx += 1
    return rows


def main() -> None:
    print("Обрабатываю 106 документов с точными страницами...")
    rows = process_our_106()
    print(f"Готово: {len(rows)} чанков с точными страницами.\n")

    print("Добавляю остальные документы из markdown Айтала (без номеров страниц)...")
    rest = process_rest_from_aital()
    rows.extend(rest)
    print(f"Добавлено: {len(rest)} чанков без привязки к странице.\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Итого: {len(rows)} чанков -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
