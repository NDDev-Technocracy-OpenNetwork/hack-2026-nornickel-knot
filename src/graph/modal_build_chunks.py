"""Параллельная сборка chunks.jsonl на Modal -- все 1232 PDF разом, не
последовательно. OCR (Tesseract) не использует GPU -- узкое место тут CPU
и число одновременных контейнеров, поэтому берём дешёвые CPU-контейнеры,
не H100, и распараллеливаем через .map().

Запуск:
  modal run src/graph/modal_build_chunks.py
"""

import json
import pathlib
import re

import modal

app = modal.App("nornikel-chunk-build")
ROOT = pathlib.Path.cwd()
PDF_ROOT = ROOT / "yandex" / "Задача 2. Научный клубок"
EXTRACTED_ROOT = ROOT / "data" / "extracted"
OUT_PATH = ROOT / "data" / "chunks.jsonl"

CHUNK_CHAR_TARGET = 2500

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("tesseract-ocr", "tesseract-ocr-rus", "tesseract-ocr-eng")
    .pip_install("pymupdf4llm==1.28.0")
    .add_local_dir(str(PDF_ROOT), "/root/pdfs")
    .add_local_dir(str(EXTRACTED_ROOT), "/root/extracted")
)


_NOISE_STRIP_RE = re.compile(
    r"<!--.*?-->|<br\s*/?>|\\|[|/@#*_~`]|^\s*Cc\}?\s*$", re.DOTALL | re.MULTILINE
)
MIN_REAL_CHARS = 25  # порог "это вообще текст, не мусор из логотипа/рамки"


def _real_content_len(text: str) -> int:
    """Символов текста после вычищения html-комментариев pymupdf4llm
    (<!-- Start/End of picture text -->), <br>, экранирующих слэшей и
    обвязки таблиц -- отличает "реально пусто" от "просто короткая
    подпись"."""
    stripped = _NOISE_STRIP_RE.sub("", text)
    stripped = re.sub(r"\s+", "", stripped)
    return len(stripped)


def _classify_chunk(text: str, real_len: int) -> str:
    table_rows = len(re.findall(r"^\s*\|.*\|\s*$", text, re.MULTILINE))
    if table_rows >= 2:
        return "table"
    if "Start of picture text" in text:
        # презентации (pptx->pdf) заворачивают ВЕСЬ текст слайда в этот же
        # маркер, не только подписи к рисункам -- длинный блок это текст
        # слайда, короткий (пара слов у картинки) это реальная подпись
        return "figure_caption" if real_len < 150 else "slide_text"
    low = text.lower()
    if "вывод" in low or "заключен" in low:
        return "conclusion"
    if "эксперимент" in low or "результат" in low:
        return "experiment"
    return "text"


def _current_section(text_before: str):
    matches = re.findall(r"^#{1,6}\s+(.+)$", text_before, re.MULTILINE)
    return matches[-1].strip("* ").strip() if matches else None


def _split_into_chunks(full_text: str, target: int = CHUNK_CHAR_TARGET):
    paragraphs = [p for p in full_text.split("\n\n") if p.strip()]
    chunks = []
    buf, buf_len, consumed = [], 0, ""
    for p in paragraphs:
        buf.append(p)
        buf_len += len(p)
        consumed += p + "\n\n"
        if buf_len >= target:
            chunks.append(("\n\n".join(buf), _current_section(consumed)))
            buf, buf_len = [], 0
    if buf:
        chunks.append(("\n\n".join(buf), _current_section(consumed)))
    return chunks


def _load_facts_by_page(stem: str) -> dict:
    matches = list(pathlib.Path("/root/extracted").rglob(f"{stem}.json"))
    if not matches:
        return {}
    doc = json.loads(matches[0].read_text(encoding="utf-8"))
    type_field = {
        "Material": "mentioned_materials", "Process": "mentioned_methods",
        "Property": "mentioned_properties", "Experiment": "mentioned_experiments",
        "Facility": "mentioned_organizations", "Expert": "mentioned_authors",
    }
    per_page: dict = {}
    for e in doc["entities"]:
        field = type_field.get(e["type"])
        if field is None:
            continue
        pg = e["provenance"]["page_start"]
        per_page.setdefault(pg, {}).setdefault(field, set()).add(e["canonical_name"])
    return per_page


def _facts_for_page(per_page: dict, page_no: int) -> dict:
    fields = per_page.get(page_no, {})
    return {k: sorted(v) for k, v in fields.items()}


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", s).strip("_")[:80]


@app.function(image=image, timeout=600, max_containers=80)
def process_one_pdf(rel_path: str) -> list[dict]:
    import pymupdf4llm

    pdf_path = pathlib.Path("/root/pdfs") / rel_path
    stem = pdf_path.stem
    slug = _slugify(stem)
    try:
        pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
    except Exception as e:  # noqa: BLE001
        return [{"document_id": stem, "error": str(e)[:200]}]

    per_page_facts = _load_facts_by_page(stem)
    rows, idx = [], 0
    last_section = None  # переносится между страницами -- заголовок раздела
    for page in pages:
        page_no = page["metadata"].get("page_number", 1)
        text = page["text"].strip()
        if not text:
            continue
        for chunk_text, section in _split_into_chunks(text):
            if section is not None:
                last_section = section
            real_len = _real_content_len(chunk_text)
            if real_len < MIN_REAL_CHARS:
                quality = "ocr_noise"
            else:
                quality = "ok"
            rows.append({
                "document_id": stem, "source_filename": pdf_path.name,
                "markdown_filename": f"{stem}.md",
                "chunk_id": f"doc_{slug}_chunk_{idx:04d}", "chunk_index": idx,
                "chunk_type": _classify_chunk(chunk_text, real_len),
                "section_title": last_section or f"page_{page_no}",
                "page_start": page_no, "page_end": page_no,
                "text": chunk_text, "token_count": len(chunk_text) // 4,
                "char_count": len(chunk_text), "real_char_count": real_len,
                "language": "ru", "extraction_quality": quality,
                "metadata": _facts_for_page(per_page_facts, page_no),
            })
            idx += 1
    return rows


@app.local_entrypoint()
def main():
    rel_paths = [str(p.relative_to(PDF_ROOT)) for p in PDF_ROOT.rglob("*.pdf")]
    print(f"Найдено {len(rel_paths)} PDF, запускаю параллельно на Modal...")

    total, failed = 0, 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i, rows in enumerate(process_one_pdf.map(rel_paths, return_exceptions=True)):
            if isinstance(rows, Exception):
                failed += 1
                print(f"  [{i+1}/{len(rel_paths)}] ОШИБКА: {rows}")
                continue
            if rows and "error" in rows[0]:
                failed += 1
                print(f"  [{i+1}/{len(rel_paths)}] {rows[0]['document_id']}: {rows[0]['error']}")
                continue
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(rows)
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(rel_paths)}] обработано, всего чанков: {total}")

    print(f"\nГотово: {total} чанков, {failed} документов упало -> {OUT_PATH.relative_to(ROOT)}")
