# Batch Ingestion Plan

This document describes the staged plan for loading 1000+ documents into the Neo4j knowledge graph.

The plan keeps the current data contract stable:

- validate `data/processed/*.jsonl`;
- load JSONL into Neo4j;
- run graph search smoke tests;
- track counts, warnings, and relation distribution after every batch.

## Shared Commands

Run these commands at each stage:

```bash
python src/validate_processed_data.py
python src/load_jsonl_graph.py --reset-db
python src/search_graph.py "<пример сущности>"
```

Example search terms:

```bash
python src/search_graph.py "операционные затраты"
python src/search_graph.py "гидрохлорирование"
python src/search_graph.py "КПП"
```

## Metrics to Track

Graph size:

- documents count;
- chunks count;
- entities count;
- mentions count;
- relations count.

Validation quality:

- CRITICAL errors;
- WARNING count;
- missing char spans;
- `surface_text_not_found`;
- `evidence_text_not_found`.

Graph semantics:

- relation type distribution;
- count of `(:Chunk)-[:MENTIONS]->(:Entity)`;
- count of semantic `(:Entity)-[r]->(:Entity)`;
- count of accidental `(:Entity)-[:MENTIONS]->(:Entity)`, which should be zero.

Useful Cypher checks:

```cypher
MATCH (d:Document)
WITH count(d) AS documents
MATCH (c:Chunk)
WITH documents, count(c) AS chunks
MATCH (e:Entity)
WITH documents, chunks, count(e) AS entities
MATCH (:Chunk)-[m:MENTIONS]->(:Entity)
WITH documents, chunks, entities, count(m) AS mentions
MATCH (:Entity)-[r]->(:Entity)
RETURN documents, chunks, entities, mentions, count(r) AS semantic_relations;
```

```cypher
MATCH (:Entity)-[r]->(:Entity)
RETURN type(r) AS relation_type, count(r) AS count
ORDER BY relation_type;
```

```cypher
MATCH (:Entity)-[r:MENTIONS]->(:Entity)
RETURN count(r) AS entity_mentions_entity_should_be_zero;
```

## Stage 1: 5 docs - current MVP

Purpose:

Confirm that the current graph contract, validator, loader, demo queries, and CLI search work end to end.

Commands:

```bash
python src/validate_processed_data.py
python src/load_jsonl_graph.py --reset-db
python src/search_graph.py "операционные затраты"
```

Expected result:

- CRITICAL errors: `0`;
- warnings may exist and should be reviewed, not treated as blockers;
- graph counts match the five JSONL files;
- source tracing works from `Document -> Chunk -> Entity`;
- semantic relations are visible between `Entity` nodes.

Decision gate:

Continue when loader summary and Cypher counts match the JSONL row counts.

## Stage 2: 20-50 docs - edge cases

Purpose:

Find contract and extraction edge cases before a larger batch.

Commands:

```bash
python src/validate_processed_data.py
python src/load_jsonl_graph.py --reset-db
python src/search_graph.py "<пример сущности>"
```

Focus areas:

- documents with missing page metadata;
- very short or empty chunks;
- duplicate names but distinct `entity_id`;
- aliases stored as list, string, or missing;
- relation types with invalid or reserved values;
- evidence text that has been normalized and no longer exactly matches chunk text.

Decision gate:

Continue when CRITICAL errors are fixed and warning categories are understood.

## Stage 3: 100 docs - stability/performance check

Purpose:

Check runtime, memory behavior, Neo4j write stability, and search usability on a medium-sized batch.

Commands:

```bash
python src/validate_processed_data.py
python src/load_jsonl_graph.py --reset-db
python src/search_graph.py "<пример сущности>"
```

Focus areas:

- loader runtime by batch size;
- constraint creation remains idempotent;
- repeated loader runs do not create duplicates because loader uses `MERGE`;
- relation type distribution remains explainable;
- search output stays readable with `--limit`, `--mentions-limit`, and `--relations-limit`.

Decision gate:

Continue when repeated reset-load runs produce stable counts and no unexpected relation types.

## Stage 4: 1000+ docs - full batch run

Purpose:

Run the full ingestion path at production-like batch size.

Commands:

```bash
python src/validate_processed_data.py
python src/load_jsonl_graph.py --reset-db
python src/search_graph.py "<пример сущности>"
```

Operational notes:

- keep `--reset-db` explicit and never clear the database by default;
- use the default `--batch-size 500` first;
- increase or decrease `--batch-size` only after measuring runtime and memory;
- keep validation enabled unless a separate validated artifact is already known to be clean;
- archive validation reports for each full run.

Decision gate:

The full run is accepted when:

- CRITICAL errors are `0`;
- warning counts are tracked and explainable;
- graph counts match the input contract;
- source tracing works for sampled entities;
- semantic relations have expected distribution;
- `Entity -> MENTIONS -> Entity` returns zero records.
