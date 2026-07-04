# Neo4j Hackathon MVP

[Русская версия](README.ru.md)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20DB-4581C3)
![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

Backend/data engineering MVP for a hackathon. The service loads structured material-science facts from JSON, validates them with Pydantic, stores them as a Neo4j knowledge graph, and exposes graph-based recommendation endpoints through FastAPI.

This repository does not perform LLM extraction yet. It expects structured JSON from an upstream LLM/ML extraction layer.

## Architecture

```text
JSON facts
↓
Pydantic validation
↓
IngestionService
↓
GraphLoader
↓
Neo4j Knowledge Graph
↓
GraphReader
↓
FastAPI API
```

Main modules:

- `src/models.py` defines the Pydantic schema for articles, experiments, methods, materials, properties, KPIs, and material facts.
- `src/parsers/json_parser.py` loads JSON facts and validates them as `MaterialFact` objects.
- `src/ingestion_service.py` coordinates ingestion.
- `src/graph_loader.py` writes nodes and relationships to Neo4j.
- `src/graph_reader.py` reads evidence paths for KPI queries.
- `src/api.py` exposes the FastAPI application.
- `main.py` loads demo data from `data/extracted/material_facts.json`.

## Graph Model

```text
Article -[:DESCRIBES]-> Experiment
Experiment -[:USES]-> Method
Experiment -[:TESTS]-> Material
Material -[:HAS_PROPERTY {value, unit}]-> Property
Property -[:AFFECTS]-> KPI
```

Main evidence path:

```text
Article -> Experiment -> Material -> Property -> KPI
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from the provided example:

```bash
cp .env.example .env
```

The default local values are:

```env
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

Start Neo4j with Docker Compose:

```bash
docker compose up -d
```

Initialize Neo4j constraints:

```bash
docker compose exec -T neo4j cypher-shell -u neo4j -p password < scripts/init_constraints.cypher
```

Load demo data:

```bash
python main.py
```

Run the API:

```bash
uvicorn src.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

### `GET /health`

Health check.

Example response:

```json
{
  "status": "ok"
}
```

### `GET /materials-for-kpi?name=Efficiency`

Returns graph evidence for materials connected to the requested KPI.

Example response:

```json
[
  {
    "article": "Conductivity study of copper alloys",
    "experiment": "Copper conductivity test",
    "method": "Lab conductivity measurement",
    "material": "Copper Alloy",
    "property": "Conductivity",
    "kpi": "Efficiency",
    "value": 0.92,
    "unit": "score",
    "evidence_path": "Conductivity study of copper alloys -> Copper conductivity test -> Copper Alloy -> Conductivity -> Efficiency"
  },
  {
    "article": "Conductivity study of aluminum alloys",
    "experiment": "Aluminum conductivity test",
    "method": "Lab conductivity measurement",
    "material": "Aluminum Alloy",
    "property": "Conductivity",
    "kpi": "Efficiency",
    "value": 0.78,
    "unit": "score",
    "evidence_path": "Conductivity study of aluminum alloys -> Aluminum conductivity test -> Aluminum Alloy -> Conductivity -> Efficiency"
  }
]
```

### `GET /recommendations?name=Efficiency`

Returns the highest-scoring material as the best candidate and includes alternatives.

Example response:

```json
{
  "kpi": "Efficiency",
  "best_candidate": {
    "material": "Copper Alloy",
    "score": 0.92,
    "reason": "Copper Alloy is linked to Efficiency through Conductivity based on Copper conductivity test from Conductivity study of copper alloys.",
    "article": "Conductivity study of copper alloys",
    "experiment": "Copper conductivity test",
    "method": "Lab conductivity measurement",
    "evidence_path": "Conductivity study of copper alloys -> Copper conductivity test -> Copper Alloy -> Conductivity -> Efficiency"
  },
  "alternatives": [
    {
      "material": "Aluminum Alloy",
      "score": 0.78,
      "property": "Conductivity",
      "evidence_path": "Conductivity study of aluminum alloys -> Aluminum conductivity test -> Aluminum Alloy -> Conductivity -> Efficiency"
    }
  ]
}
```

## Why Neo4j

Neo4j is used because the main value of this MVP is relationship traversal. Material recommendations are not based only on isolated rows; they depend on explainable paths from articles and experiments to materials, properties, and KPIs. A graph database makes those paths explicit, queryable, and easy to return as evidence.

## Current Status

Implemented:

- JSON loading from `data/extracted/material_facts.json`.
- Pydantic validation for structured material facts.
- Neo4j node and relationship ingestion.
- Uniqueness constraints for graph identifiers.
- Evidence path retrieval for KPI-based material queries.
- FastAPI endpoints for health checks, KPI material lookup, and simple recommendations.

Not implemented yet:

- LLM/ML extraction from papers or raw text.
- Authentication or authorization.
- Automated tests.
- Ranking beyond sorting by the stored relationship value.
- Production deployment configuration.

## Hackathon Role

This repository is the Data/Graph Engineering layer.

Upstream, an LLM/ML extraction layer is expected to produce structured JSON facts. This service handles schema validation, graph ingestion, evidence path retrieval, and API responses for downstream product or demo interfaces.
