"""Собирает выгрузку строго под "Data Contract for PDF-DOCX to Markdown
Pipeline" (документ от Данилы) -- единый document_id связывает всё:
documents.jsonl, chunks.jsonl, entities.jsonl, entity_mentions.jsonl,
relations.jsonl, harmonic_documents.jsonl, extraction_report.csv.

document_id -- слаг из имени файла (не порядковый doc_001), потому что
контракт требует стабильности ID при повторной обработке ОДНОГО документа:
слаг стабилен независимо от того, в каком порядке и какой подмножество
документов обрабатывается, порядковый номер -- нет.

Источники:
  data/parsed/*.json    -- сырой текст по страницам (для pages_count)
  data/extracted/*.json -- сущности и связи (наш текущий формат id-based)
  data/chunks.jsonl      -- чанки, собранные ранее (src/graph/modal_build_chunks.py)
  data/harmonic_runs/doc_embeddings.pt -- эмбеддинги документов

Запуск:
  uv run python3 src/graph/build_data_contract.py
"""

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PARSED_ROOT = ROOT / "data" / "parsed"
EXTRACTED_ROOT = ROOT / "data" / "extracted"
CHUNKS_JSONL = ROOT / "data" / "chunks.jsonl"
DOC_EMBEDDINGS = ROOT / "data" / "harmonic_runs" / "doc_embeddings.pt"
OUT_DIR = ROOT / "data" / "contract"

SCHEMA_VERSION = "1.0"
PIPELINE_RUN_ID = "run_" + datetime.now(timezone.utc).strftime("%Y_%m_%d_001")
NOW = datetime.now(timezone.utc).isoformat()

# наш relation -> словарь Данилы (минимальный набор из контракта)
RELATION_MAP = {
    "uses_material": "USES",
    "operates_at_condition": "AFFECTS",
    "produces_output": "PRODUCES",
    "described_in": "EXTRACTED_FROM",
    "validated_by": "SUPPORTS",
    "contradicts": "CONTRADICTS",
    "affiliated_with": "BELONGS_TO",
}

# наш entity type -> словарь Данилы
ENTITY_TYPE_MAP = {
    "Material": "Material", "Process": "Method", "Equipment": "Equipment",
    "Property": "Property", "Experiment": "Experiment", "Publication": "Document",
    "Expert": "Expert", "Facility": "Organization",
}


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", s).strip("_")[:80]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def entity_id(entity_type: str, canonical_name: str) -> str:
    mapped_type = ENTITY_TYPE_MAP.get(entity_type, "Unknown")
    return f"{mapped_type.lower()}_{slugify(canonical_name)}"


def load_parsed_pages() -> dict:
    out = {}
    for p in PARSED_ROOT.rglob("*.json"):
        doc = json.loads(p.read_text(encoding="utf-8"))
        out[doc["filename"]] = doc
    return out


def load_chunk_page_index() -> dict:
    """document_id -> отсортированный список (page_start, page_end, chunk_id,
    text) из НОВОЙ (page-aware) нарезки chunks.jsonl. Нужен, чтобы у
    сущностей и связей (которые несут page_start/page_end от старого
    извлечения) был chunk_id, реально указывающий на текст в chunks.jsonl,
    а не на случайно совпавшую по имени, но другую по содержимому старую
    нарезку."""
    index: dict[str, list] = {}
    if not CHUNKS_JSONL.exists():
        return index
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            stem = row["document_id"]
            document_id = f"doc_{slugify(stem)}"
            if row.get("page_start") is None:
                continue
            index.setdefault(document_id, []).append(
                (row["page_start"], row["page_end"], row["chunk_id"], row["text"])
            )
    for doc_id in index:
        index[doc_id].sort()
    return index


def load_doc_structure_flags() -> dict:
    """document_id -> {has_tables, has_figures} по факту наличия чанков
    соответствующего типа -- чтобы documents.jsonl не нёс null там, где
    ответ реально известен из уже собранных чанков."""
    flags: dict[str, dict] = {}
    if not CHUNKS_JSONL.exists():
        return flags
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            document_id = f"doc_{slugify(row['document_id'])}"
            rec = flags.setdefault(document_id, {"has_tables": False, "has_figures": False})
            if row["chunk_type"] == "table":
                rec["has_tables"] = True
            if row["chunk_type"] == "figure_caption":
                rec["has_figures"] = True
    return flags


def find_chunk_id(page_index: dict, document_id: str, page: int, surface_text: str = "") -> str | None:
    """Если на странице несколько чанков, предпочитаем тот, где реально
    встречается текст сущности/факта -- иначе просто первый по странице
    (лучше приблизительно верная страница, чем ничего)."""
    candidates = [
        (chunk_id, text) for page_start, page_end, chunk_id, text in page_index.get(document_id, [])
        if page_start <= page <= page_end
    ]
    if not candidates:
        return None
    if surface_text:
        for chunk_id, text in candidates:
            if surface_text.lower() in text.lower():
                return chunk_id
    return candidates[0][0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parsed_by_filename = load_parsed_pages()

    documents_f = open(OUT_DIR / "documents.jsonl", "w", encoding="utf-8")
    entities_f = open(OUT_DIR / "entities.jsonl", "w", encoding="utf-8")
    mentions_f = open(OUT_DIR / "entity_mentions.jsonl", "w", encoding="utf-8")
    relations_f = open(OUT_DIR / "relations.jsonl", "w", encoding="utf-8")
    report_f = open(OUT_DIR / "extraction_report.csv", "w", encoding="utf-8", newline="")
    report_writer = csv.writer(report_f)
    report_writer.writerow([
        "pipeline_run_id", "document_id", "source_filename", "status", "pages_count",
        "chunks_count", "entities_count", "relations_count", "has_ocr", "has_tables",
        "has_figures", "extraction_quality", "warnings", "errors", "needs_manual_review",
        "created_at",
    ])

    global_entities: dict[str, dict] = {}  # entity_id -> запись (пишем в конце, без дублей)
    mention_counter = 0
    relation_counter = 0
    filename_to_docid: dict[str, str] = {}
    page_index = load_chunk_page_index()
    doc_structure_flags = load_doc_structure_flags()
    unmatched_chunk_refs = 0

    extracted_files = sorted(EXTRACTED_ROOT.rglob("*.json"))
    for ex_path in extracted_files:
        doc = json.loads(ex_path.read_text(encoding="utf-8"))
        filename = doc["filename"]
        stem = Path(filename).stem
        document_id = f"doc_{slugify(stem)}"
        filename_to_docid[filename] = document_id

        parsed = parsed_by_filename.get(filename)
        pages_count = parsed["num_pages"] if parsed else None
        content_hash = sha256(filename + doc["category"])

        struct_flags = doc_structure_flags.get(document_id, {"has_tables": None, "has_figures": None})
        documents_f.write(json.dumps({
            "schema_version": SCHEMA_VERSION, "pipeline_run_id": PIPELINE_RUN_ID,
            "document_id": document_id, "source_filename": filename,
            "source_path": f"data/raw/{doc['category']}/{filename}",
            "markdown_filename": None, "markdown_path": None,
            "title": stem, "document_type": "report", "language": "ru",
            "pages_count": pages_count, "content_hash": content_hash,
            "parser_name": "pymupdf_raw", "parsing_method": "raw_text_extraction",
            "has_tables": struct_flags["has_tables"], "has_figures": struct_flags["has_figures"],
            "has_ocr": bool(parsed and parsed.get("likely_scanned")),
            "extraction_quality": "ok", "created_at": NOW,
        }, ensure_ascii=False) + "\n")

        local_to_entity_id: dict[str, str] = {}
        for e in doc["entities"]:
            eid = entity_id(e["type"], e["canonical_name"])
            local_to_entity_id[e["id"]] = eid
            mapped_type = ENTITY_TYPE_MAP.get(e["type"], "Unknown")

            if eid not in global_entities:
                global_entities[eid] = {
                    "schema_version": SCHEMA_VERSION, "entity_id": eid, "canonical_id": eid,
                    "entity_type": mapped_type, "name": e["canonical_name"],
                    "normalized_name": e["canonical_name"], "aliases": set(),
                    "description": None, "confidence": e.get("confidence") or 0.8,
                }
            global_entities[eid]["aliases"].add(e["name"])

            mention_counter += 1
            prov = e["provenance"]
            real_chunk_id = find_chunk_id(page_index, document_id, prov["page_start"], e["name"])
            if real_chunk_id is None:
                unmatched_chunk_refs += 1
                real_chunk_id = f"{stem}::chunk{prov['chunk_index']}"  # fallback, старая нарезка
            mentions_f.write(json.dumps({
                "schema_version": SCHEMA_VERSION, "mention_id": f"mention_{mention_counter:06d}",
                "entity_id": eid, "entity_type": mapped_type, "surface_text": e["name"],
                "document_id": document_id, "chunk_id": real_chunk_id,
                "page_start": prov["page_start"], "page_end": prov["page_end"],
                "char_start": None, "char_end": None, "confidence": e.get("confidence") or 0.8,
            }, ensure_ascii=False) + "\n")

        for r in doc["relations"]:
            src_eid = local_to_entity_id.get(r["source_id"])
            tgt_eid = local_to_entity_id.get(r["target_id"])
            if src_eid is None or tgt_eid is None:
                continue
            relation_counter += 1
            prov = r["provenance"]
            evidence = f"{global_entities[src_eid]['name']} {r['relation']} {global_entities[tgt_eid]['name']}"
            if r.get("numeric_value") is not None:
                evidence += f" (value={r['numeric_value']}{r.get('unit') or ''})"
            real_chunk_id = find_chunk_id(
                page_index, document_id, prov["page_start"], global_entities[src_eid]["name"]
            )
            if real_chunk_id is None:
                unmatched_chunk_refs += 1
                real_chunk_id = f"{stem}::chunk{prov['chunk_index']}"  # fallback, старая нарезка
            relations_f.write(json.dumps({
                "schema_version": SCHEMA_VERSION, "relation_id": f"rel_{relation_counter:06d}",
                "source_entity_id": src_eid, "source_entity_type": global_entities[src_eid]["entity_type"],
                "target_entity_id": tgt_eid, "target_entity_type": global_entities[tgt_eid]["entity_type"],
                "relation_type": RELATION_MAP.get(r["relation"], "RELATED_TO"),
                "document_id": document_id, "chunk_id": real_chunk_id,
                "evidence_text": evidence, "page_start": prov["page_start"], "page_end": prov["page_end"],
                "confidence": r.get("confidence") or 0.75, "extraction_method": "llm",
                "created_at": NOW,
            }, ensure_ascii=False) + "\n")

        report_writer.writerow([
            PIPELINE_RUN_ID, document_id, filename, "success", pages_count or "",
            "", len(doc["entities"]), len(doc["relations"]),
            bool(parsed and parsed.get("likely_scanned")), "", "", "ok", "", "", False, NOW,
        ])

    for rec in global_entities.values():
        rec["aliases"] = sorted(rec["aliases"])
        entities_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    documents_f.close()
    entities_f.close()
    mentions_f.close()
    relations_f.close()
    report_f.close()

    # chunks.jsonl -- перекладываем из уже собранного data/chunks.jsonl под
    # точные имена полей контракта, добавляя document_id по имени файла
    chunks_out = open(OUT_DIR / "chunks.jsonl", "w", encoding="utf-8")
    n_chunks = 0
    if CHUNKS_JSONL.exists():
        with open(CHUNKS_JSONL, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                stem = row["document_id"]
                document_id = filename_to_docid.get(row["source_filename"], f"doc_{slugify(stem)}")
                text = row["text"]
                chunks_out.write(json.dumps({
                    "schema_version": SCHEMA_VERSION, "pipeline_run_id": PIPELINE_RUN_ID,
                    "document_id": document_id, "source_filename": row["source_filename"],
                    "chunk_id": row["chunk_id"],
                    "chunk_index": row["chunk_index"], "chunk_type": row["chunk_type"],
                    "section_title": row.get("section_title"), "section_path": None,
                    "page_start": row.get("page_start"), "page_end": row.get("page_end"),
                    "char_start": None, "char_end": None, "text": text,
                    "text_hash": sha256(text), "token_count": row["token_count"],
                    "char_count": row["char_count"],
                    "real_char_count": row.get("real_char_count"),
                    "has_table": row["chunk_type"] == "table",
                    "has_figure": row["chunk_type"] == "figure_caption",
                    "overlap_prev": False, "overlap_next": False,
                    "extraction_quality": row.get("extraction_quality", "ok"),
                }, ensure_ascii=False) + "\n")
                n_chunks += 1
    chunks_out.close()

    # harmonic_documents.jsonl
    harmonic_out = open(OUT_DIR / "harmonic_documents.jsonl", "w", encoding="utf-8")
    n_harmonic = 0
    if DOC_EMBEDDINGS.exists():
        entries = torch.load(DOC_EMBEDDINGS, weights_only=False)
        for e in entries:
            document_id = filename_to_docid.get(e["filename"], f"doc_{slugify(Path(e['filename']).stem)}")
            harmonic_out.write(json.dumps({
                "schema_version": SCHEMA_VERSION, "pipeline_run_id": PIPELINE_RUN_ID,
                "document_id": document_id, "source_filename": e["filename"],
                "harmonic_document_id": f"harmonic_{document_id}",
                "harmonic_index_name": "nornikel_corpus_v1", "embedding_model": "harmonic",
                "embedding_dim": len(e["vector"]), "input_level": "document",
                "input_text_hash": sha256(e["filename"]), "created_at": NOW,
            }, ensure_ascii=False) + "\n")
            n_harmonic += 1
    harmonic_out.close()

    print(f"documents: {len(extracted_files)}")
    print(f"entities (уникальных): {len(global_entities)}")
    print(f"entity_mentions: {mention_counter}")
    print(f"relations: {relation_counter}")
    print(f"chunks: {n_chunks}")
    print(f"harmonic_documents: {n_harmonic}")
    print(f"chunk_id без точного совпадения по странице (fallback на старую нарезку): {unmatched_chunk_refs}")
    print(f"\nВсё в {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
