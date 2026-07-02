"""Parses PDFs in data/sample/ into plain text + metadata JSON in data/parsed/.

Flags documents with very little extractable text per page as likely_scanned
(no text layer) -- those need OCR, which is out of scope for tonight's MVP.
"""

import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "data" / "sample"
DEST_ROOT = ROOT / "data" / "parsed"

MIN_CHARS_PER_PAGE = 40  # below this average, treat as scanned/no text layer


def parse_pdf(path: Path) -> dict:
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()

    char_count = sum(len(p["text"]) for p in pages)
    avg_chars_per_page = char_count / max(len(pages), 1)

    return {
        "filename": path.name,
        "category": path.parent.name,
        "num_pages": len(pages),
        "char_count": char_count,
        "likely_scanned": avg_chars_per_page < MIN_CHARS_PER_PAGE,
        "pages": pages,
    }


def main() -> None:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(SOURCE_ROOT.rglob("*.pdf"))
    scanned = []

    for pdf_path in pdfs:
        category = pdf_path.parent.name
        dest_dir = DEST_ROOT / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / (pdf_path.stem + ".json")

        parsed = parse_pdf(pdf_path)
        dest_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

        flag = " [SCANNED?]" if parsed["likely_scanned"] else ""
        print(f"{parsed['filename']:<45} pages={parsed['num_pages']:<4} chars={parsed['char_count']:<8}{flag}")
        if parsed["likely_scanned"]:
            scanned.append(parsed["filename"])

    print(f"\nParsed {len(pdfs)} files into {DEST_ROOT.relative_to(ROOT)}")
    if scanned:
        print(f"Likely scanned (no usable text layer, {len(scanned)}): {scanned}")


if __name__ == "__main__":
    main()
