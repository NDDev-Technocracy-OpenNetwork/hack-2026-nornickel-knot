# Data Contract v1

This document describes the current `data/processed/*.jsonl` contract used by the Neo4j MVP loader.

The graph model is:

- `documents.jsonl` -> `(:Document)`
- `chunks.jsonl` -> `(:Chunk)` and `(:Document)-[:CONTAINS]->(:Chunk)`
- `entities.jsonl` -> `(:Entity)`
- `entity_mentions.jsonl` -> `(:Chunk)-[:MENTIONS]->(:Entity)`
- `relations.jsonl` -> semantic `(:Entity)-[:RELATION_TYPE]->(:Entity)`

## Key Identifiers

- `document_id` links `Document` and `Chunk`.
- `chunk_id` links `Chunk`, `Mention`, and `Relation`.
- `entity_id` links `Entity`, `Mention`, and `Relation`.
- `relation_id` identifies one semantic relation row in `relations.jsonl`.

## documents.jsonl

Purpose:

`documents.jsonl` defines source documents. Each row becomes one Neo4j `Document` node.

Main fields:

- `schema_version`
- `pipeline_run_id`
- `document_id`
- `source_filename`
- `source_path`
- `markdown_filename`
- `markdown_path`
- `title`
- `document_type`
- `language`
- `pages_count`
- `content_hash`
- `parser_name`
- `parsing_method`
- `has_tables`
- `has_figures`
- `has_ocr`
- `extraction_quality`
- `created_at`

Required fields:

- `document_id`

Nullable fields:

- Metadata fields may be nullable if the upstream parser cannot provide them, but `document_id` must be present and non-empty.

Links to other files:

- `chunks.jsonl.document_id` must reference an existing `documents.jsonl.document_id`.

Neo4j loading:

```cypher
MERGE (d:Document {document_id: row.document_id})
SET d += row
```

## chunks.jsonl

Purpose:

`chunks.jsonl` defines source text chunks extracted from documents. Each row becomes one Neo4j `Chunk` node and is connected to its source `Document`.

Main fields:

- `schema_version`
- `pipeline_run_id`
- `document_id`
- `chunk_id`
- `chunk_index`
- `chunk_type`
- `section_title`
- `section_path`
- `page_start`
- `page_end`
- `char_start`
- `char_end`
- `text`
- `text_hash`
- `token_count`
- `char_count`
- `has_table`
- `has_figure`
- `overlap_prev`
- `overlap_next`
- `extraction_quality`

Required fields:

- `chunk_id`
- `document_id`
- `text`

Nullable fields:

- `section_title`
- `page_start`
- `page_end`
- `char_start`
- `char_end`

Links to other files:

- `document_id` must reference `documents.jsonl.document_id`.
- `entity_mentions.jsonl.chunk_id` references `chunks.jsonl.chunk_id`.
- `relations.jsonl.chunk_id` references `chunks.jsonl.chunk_id`.

Neo4j loading:

```cypher
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c += row
WITH c, row
MATCH (d:Document {document_id: row.document_id})
MERGE (d)-[:CONTAINS]->(c)
```

## entities.jsonl

Purpose:

`entities.jsonl` defines normalized domain entities. Each row becomes one Neo4j `Entity` node.

Main fields:

- `schema_version`
- `entity_id`
- `entity_type`
- `name`
- `normalized_name`
- `aliases`
- `description`
- `confidence`

Required fields:

- `entity_id`
- `entity_type`
- `name`

Nullable fields:

- `normalized_name`
- `aliases`
- `description`
- `confidence`

Links to other files:

- `entity_mentions.jsonl.entity_id` references `entities.jsonl.entity_id`.
- `relations.jsonl.source_entity_id` references `entities.jsonl.entity_id`.
- `relations.jsonl.target_entity_id` references `entities.jsonl.entity_id`.

Neo4j loading:

```cypher
MERGE (e:Entity {entity_id: row.entity_id})
SET e += row
```

## entity_mentions.jsonl

Purpose:

`entity_mentions.jsonl` links source chunks to normalized entities. It is the source-tracing layer of the graph.

Main fields:

- `schema_version`
- `mention_id`
- `entity_id`
- `entity_type`
- `surface_text`
- `document_id`
- `chunk_id`
- `page_start`
- `page_end`
- `char_start`
- `char_end`
- `confidence`

Required fields:

- `mention_id`
- `entity_id`
- `chunk_id`

Nullable fields:

- `surface_text`
- `page_start`
- `page_end`
- `char_start`
- `char_end`
- `confidence`

Links to other files:

- `chunk_id` must reference `chunks.jsonl.chunk_id`.
- `entity_id` must reference `entities.jsonl.entity_id`.
- `document_id` should match the document owning the referenced chunk.

Neo4j loading:

```cypher
MATCH (c:Chunk {chunk_id: row.chunk_id})
MATCH (e:Entity {entity_id: row.entity_id})
MERGE (c)-[m:MENTIONS {mention_id: row.mention_id}]->(e)
SET m += row
```

`MENTIONS` is reserved for `(:Chunk)-[:MENTIONS]->(:Entity)` source tracing.

## relations.jsonl

Purpose:

`relations.jsonl` defines semantic relations between normalized entities. Each row becomes one typed Neo4j relationship from source `Entity` to target `Entity`.

Main fields:

- `schema_version`
- `relation_id`
- `source_entity_id`
- `source_entity_type`
- `target_entity_id`
- `target_entity_type`
- `relation_type`
- `document_id`
- `chunk_id`
- `evidence_text`
- `page_start`
- `page_end`
- `confidence`
- `extraction_method`
- `created_at`

Required fields:

- `relation_id`
- `source_entity_id`
- `target_entity_id`
- `relation_type`
- `chunk_id`

Nullable fields:

- `evidence_text`
- `page_start`
- `page_end`
- `confidence`
- `extraction_method`
- `created_at`

Links to other files:

- `chunk_id` must reference `chunks.jsonl.chunk_id`.
- `source_entity_id` must reference `entities.jsonl.entity_id`.
- `target_entity_id` must reference `entities.jsonl.entity_id`.
- `document_id` should match the document owning the referenced chunk.

Neo4j loading:

```cypher
MATCH (src:Entity {entity_id: row.source_entity_id})
MATCH (dst:Entity {entity_id: row.target_entity_id})
MERGE (src)-[r:RELATION_TYPE {relation_id: row.relation_id}]->(dst)
SET r += row
```

`relation_type` is normalized by the loader. Empty, invalid, or reserved `MENTIONS` values are loaded as `RELATED_TO`.

## Validation Rules

The validator checks:

- all five files exist;
- every file is readable as JSONL;
- `document_id`, `chunk_id`, `entity_id`, `mention_id`, and `relation_id` are unique;
- chunk, mention, and relation references point to existing rows;
- `relation_type` is non-empty;
- `confidence`, if present, is numeric and between `0` and `1`;
- warning-level quality checks for empty text, missing spans, missing evidence, and source text mismatches.

## NER is not a blocker

NER is not required to build the current graph.

The current graph already builds from existing `entities.jsonl`, `entity_mentions.jsonl`, and `relations.jsonl` files. The loader can create `Document`, `Chunk`, `Entity`, source-tracing `MENTIONS`, and semantic `Entity -> Entity` relationships from the current contract.

NER is a quality improvement for `entity_mentions.jsonl`, not a blocker for ingestion. Better NER helps:

- find more accurate `surface_text`;
- fill `char_start` and `char_end` spans more reliably;
- improve `aliases`;
- improve `normalized_name`;
- reduce warning counts such as `surface_text_not_found` and missing char spans.

Semantic relations already live in `relations.jsonl` and are validated separately. They do not depend on rerunning NER as long as their `source_entity_id`, `target_entity_id`, `chunk_id`, `relation_id`, and `relation_type` satisfy the contract.
