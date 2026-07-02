"""Expands data/sample/ from the original 18 hand-picked docs to a larger
cross-section (~180 docs) of the corpus, still PDF-only with a real text
layer (checked directly with PyMuPDF, not guessed from file size).

Used for: (1) a bigger Cerebras fact-extraction batch, (2) more raw text for
the Harmonic training corpus -- parse.py keeps full per-page text regardless
of what extract.py later caps for API calls.
"""

import random
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "yandex" / "Задача 2. Научный клубок" / "Источники информации"
DEST_ROOT = ROOT / "data" / "sample"

CATEGORIES = ["Доклады", "Журналы", "Материалы конференций", "Обзоры"]
TARGET_PER_CATEGORY = 45  # ~180 total across 4 categories
MIN_CHARS_PER_PAGE = 40
MAX_FILE_MB = 15  # skip huge scans/image-heavy PDFs, keep parsing fast
SEED = 42


def has_text_layer(path: Path) -> bool:
    try:
        doc = pymupdf.open(path)
        pages_to_check = min(3, len(doc))
        if pages_to_check == 0:
            return False
        chars = sum(len(doc[i].get_text("text")) for i in range(pages_to_check))
        doc.close()
        return (chars / pages_to_check) >= MIN_CHARS_PER_PAGE
    except Exception:
        return False


def main() -> None:
    rng = random.Random(SEED)
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    already = {p.name for p in DEST_ROOT.rglob("*.pdf")}
    total_copied = 0

    for category in CATEGORIES:
        cat_dir = SOURCE_ROOT / category
        candidates = [
            p for p in cat_dir.rglob("*.pdf")
            if p.name not in already and p.stat().st_size <= MAX_FILE_MB * 1024 * 1024
        ]
        rng.shuffle(candidates)

        dest_dir = DEST_ROOT / category
        dest_dir.mkdir(parents=True, exist_ok=True)

        copied_here = len([p for p in dest_dir.glob("*.pdf")])
        checked = 0
        for cand in candidates:
            if copied_here >= TARGET_PER_CATEGORY:
                break
            checked += 1
            if not has_text_layer(cand):
                continue
            dest = dest_dir / cand.name
            dest.write_bytes(cand.read_bytes())
            copied_here += 1
            total_copied += 1

        print(f"{category:<25} now {copied_here} files (checked {checked} candidates)")

    print(f"\nTotal PDFs in data/sample now: {len(list(DEST_ROOT.rglob('*.pdf')))}")


if __name__ == "__main__":
    main()
