# AGENTS.md

## Project Purpose

`neo4j-hackathon-mvp` is an R&D knowledge graph ingestion pipeline for Neo4j.

The current production-oriented flow loads normalized JSONL files from `data/processed` into Neo4j using `src/load_jsonl_graph.py`.

## Data Layout

`data/processed` contains the graph import contract:

- `data/processed/documents.jsonl` -> `(:Document)`
- `data/processed/chunks.jsonl` -> `(:Chunk)` and `(:Document)-[:CONTAINS]->(:Chunk)`
- `data/processed/entities.jsonl` -> `(:Entity)`
- `data/processed/entity_mentions.jsonl` -> `(:Chunk)-[:MENTIONS]->(:Entity)`
- `data/processed/relations.jsonl` -> `(:Entity)-[:RELATION_TYPE]->(:Entity)`

## Commands

Start Neo4j:

```bash
docker compose up -d
```

Initialize constraints:

```bash
docker compose exec -T neo4j cypher-shell -u neo4j -p password < scripts/init_constraints.cypher
```

Run the JSONL graph loader:

```bash
python src/load_jsonl_graph.py
```

Run the reset-and-load flow only when explicitly requested:

```bash
python src/load_jsonl_graph.py --reset-db
```

Run the processed-data validator if `src/validate_processed_data.py` exists:

```bash
python src/validate_processed_data.py
```

## Working Rules

- Do not delete, truncate, regenerate, or rewrite `data/processed` without a separate explicit request.
- Do not delete or clear the Neo4j database unless the user explicitly passes or requests a `--reset-db` flow.
- Do not rewrite working code from scratch.
- Make small, focused changes.
- Do not add new dependencies without explaining why the existing stack is insufficient.
- After changing Python code, run the corresponding script or focused validation command.
- After making changes, show `git diff` so the user can review the exact patch.
- The loader must use `MERGE` so repeated runs do not create duplicate nodes or relationships.
- Keep the JSONL schema backward-compatible.
