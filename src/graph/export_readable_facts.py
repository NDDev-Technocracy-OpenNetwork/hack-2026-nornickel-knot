"""Человекочитаемая версия извлечённых фактов: связи по id (source_id/
target_id) не читаются глазами без сверки с массивом entities. Тут вместо
id сразу подставлены имена и типы сущностей прямо в каждую связь.

Вход:  data/extracted/*.json (id-based, как есть)
Выход: data/extracted_readable/*.json (имена прямо в relations)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "data" / "extracted"
DEST_ROOT = ROOT / "data" / "extracted_readable"


def main() -> None:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0

    for doc_path in sorted(SOURCE_ROOT.rglob("*.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        by_id = {e["id"]: e for e in doc["entities"]}

        readable_relations = []
        for r in doc["relations"]:
            src = by_id.get(r["source_id"])
            tgt = by_id.get(r["target_id"])
            if src is None or tgt is None:
                continue
            readable_relations.append({
                "fact": f"{src['canonical_name']} --[{r['relation']}]--> {tgt['canonical_name']}",
                "source_name": src["canonical_name"], "source_type": src["type"],
                "relation": r["relation"],
                "target_name": tgt["canonical_name"], "target_type": tgt["type"],
                "numeric_value": r.get("numeric_value"), "unit": r.get("unit"),
                "condition": r.get("condition"), "confidence": r.get("confidence"),
                "source_file": r["provenance"]["source_file"],
                "page_start": r["provenance"]["page_start"],
                "page_end": r["provenance"]["page_end"],
            })
        total += len(readable_relations)

        rel_dir = doc_path.parent.name
        dest_dir = DEST_ROOT / rel_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / doc_path.name).write_text(
            json.dumps({
                "filename": doc["filename"], "category": doc["category"],
                "entities": doc["entities"], "facts": readable_relations,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Готово: {total} читаемых фактов -> {DEST_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
