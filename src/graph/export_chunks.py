"""Отдельный JSON-слой "чанков" -- Данила (Neo4j) попросил явный файл
с содержимым каждого куска текста, а не только фактов из него. Раньше
чанки жили только в памяти на время LLM-вызова и выбрасывались.

Выход: data/chunks/<category>/<filename>.json -- список чанков документа с
{chunk_id, document_id, chunk_index, page_start, page_end, content}.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "extract"))
from extract import MAX_CHARS_PER_DOC, CHUNK_SIZE, make_chunks  # noqa: E402

SOURCE_ROOT = ROOT / "data" / "parsed"
DEST_ROOT = ROOT / "data" / "chunks"


def main() -> None:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    total_chunks = 0

    for parsed_path in sorted(SOURCE_ROOT.rglob("*.json")):
        doc = json.loads(parsed_path.read_text(encoding="utf-8"))
        document_id = doc["filename"]
        chunks = make_chunks(doc["pages"], MAX_CHARS_PER_DOC, CHUNK_SIZE)

        rows = [
            {
                "chunk_id": f"{document_id}::chunk{i}",
                "document_id": document_id,
                "category": doc["category"],
                "chunk_index": i,
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "content": c["text"],
            }
            for i, c in enumerate(chunks)
        ]
        total_chunks += len(rows)

        dest_dir = DEST_ROOT / doc["category"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / parsed_path.name).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Готово: {total_chunks} чанков -> {DEST_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
