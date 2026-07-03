"""Экспортирует уже извлечённые факты (data/extracted/*.json) в Neo4j.

Не пересчитывает ничего заново -- берёт тот же граф, что и наш networkx-путь
(см. graph_store.build_graph), просто льёт в Neo4j вместо pickle.

Запуск:
  uv add neo4j
  NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=... \
    uv run python3 src/graph/export_neo4j.py

Схема:
  Узел: (:Material|Process|Equipment|Property|Experiment|Publication|Expert|Facility
          {canonical_name, surface_forms, mentions})
  Связь: [:USES_MATERIAL|OPERATES_AT_CONDITION|PRODUCES_OUTPUT|DESCRIBED_IN|
           VALIDATED_BY|CONTRADICTS|AFFILIATED_WITH
          {source_file, category, page_start, page_end, numeric_value, unit,
           condition, confidence}]
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "data" / "extracted"


def node_key(entity_type: str, canonical_name: str) -> str:
    return f"{entity_type}:{canonical_name.strip().lower()}"


def load_facts() -> tuple[dict[str, dict], list[dict]]:
    """Тот же процесс сборки, что и graph_store.build_graph, но в виде
    плоских списков узлов/связей, удобных для батч-заливки в Neo4j."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    for doc_path in sorted(SOURCE_ROOT.rglob("*.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        local_to_key: dict[str, str] = {}

        for e in doc["entities"]:
            key = node_key(e["type"], e["canonical_name"])
            local_to_key[e["id"]] = key
            if key in nodes:
                nodes[key]["surface_forms"].add(e["name"])
                nodes[key]["mentions"] += 1
                nodes[key]["sources"].add(e["provenance"]["source_file"])
            else:
                nodes[key] = {
                    "type": e["type"],
                    "canonical_name": e["canonical_name"],
                    "surface_forms": {e["name"]},
                    "mentions": 1,
                    "sources": {e["provenance"]["source_file"]},
                }

        for r in doc["relations"]:
            src = local_to_key.get(r["source_id"])
            tgt = local_to_key.get(r["target_id"])
            if src is None or tgt is None or src not in nodes or tgt not in nodes:
                continue
            edges.append({
                "source": src, "target": tgt, "relation": r["relation"],
                "numeric_value": r.get("numeric_value"), "unit": r.get("unit"),
                "condition": r.get("condition"), "confidence": r.get("confidence"),
                "source_file": r["provenance"]["source_file"],
                "category": r["provenance"]["category"],
                "page_start": r["provenance"]["page_start"],
                "page_end": r["provenance"]["page_end"],
            })

    return nodes, edges


def main() -> None:
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ["NEO4J_PASSWORD"]

    nodes, edges = load_facts()
    print(f"Собрано локально: {len(nodes)} узлов, {len(edges)} связей")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("CREATE CONSTRAINT node_key IF NOT EXISTS "
                     "FOR (n:Entity) REQUIRE n.key IS UNIQUE")

        # Без APOC (может не стоять на инстансе Данилы): label и тип связи
        # нельзя параметризовать в Cypher напрямую, поэтому узлы получают
        # общий :Entity + точечный MATCH...SET per тип, а связи -- отдельный
        # CREATE per тип связи (UNWIND+MERGE тут не подходит: мультиграф,
        # между одной парой узлов бывает несколько разных фактов).
        node_rows = [
            {"key": k, "type": v["type"], "canonical_name": v["canonical_name"],
             "surface_forms": list(v["surface_forms"]), "mentions": v["mentions"],
             "sources": list(v["sources"])}
            for k, v in nodes.items()
        ]
        session.run(
            "UNWIND $rows AS row MERGE (n:Entity {key: row.key}) "
            "SET n.type = row.type, n.canonical_name = row.canonical_name, "
            "n.surface_forms = row.surface_forms, n.mentions = row.mentions, "
            "n.sources = row.sources",
            rows=node_rows,
        )
        for etype in {v["type"] for v in nodes.values()}:
            session.run(f"MATCH (n:Entity {{type: $t}}) SET n:`{etype}`", t=etype)
        print(f"Узлы залиты: {len(node_rows)}")

        by_relation: dict[str, list] = {}
        for e in edges:
            by_relation.setdefault(e["relation"].upper(), []).append(e)
        for rel_type, rows in by_relation.items():
            session.run(
                f"""
                UNWIND $rows AS row
                MATCH (a:Entity {{key: row.source}}), (b:Entity {{key: row.target}})
                CREATE (a)-[r:`{rel_type}` {{
                    numeric_value: row.numeric_value, unit: row.unit,
                    condition: row.condition, confidence: row.confidence,
                    source_file: row.source_file, category: row.category,
                    page_start: row.page_start, page_end: row.page_end
                }}]->(b)
                """,
                rows=rows,
            )
            print(f"  {rel_type}: {len(rows)} связей залито")

    driver.close()
    print("Готово.")


if __name__ == "__main__":
    main()
