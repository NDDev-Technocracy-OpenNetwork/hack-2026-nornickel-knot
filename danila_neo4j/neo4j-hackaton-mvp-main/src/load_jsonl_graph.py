import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_relation_type(value: str) -> str:
    if not value:
        return "RELATED_TO"
    value = str(value).upper().strip()
    if value == "MENTIONS" or not re.fullmatch(r"[A-Z_]+", value):
        return "RELATED_TO"
    return value


def run_validator(data_dir: Path):
    validator_path = Path(__file__).with_name("validate_processed_data.py")
    command = [
        sys.executable,
        str(validator_path),
        "--data-dir",
        str(data_dir),
    ]
    print("Running processed data validator...", flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"Validation failed with exit code {result.returncode}. Loading stopped.")
        sys.exit(result.returncode)
    print("Validation passed. Continuing with graph load.")


def create_constraints(tx):
    tx.run("""
    CREATE CONSTRAINT document_id_unique IF NOT EXISTS
    FOR (d:Document)
    REQUIRE d.document_id IS UNIQUE
    """)
    tx.run("""
    CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
    FOR (c:Chunk)
    REQUIRE c.chunk_id IS UNIQUE
    """)
    tx.run("""
    CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
    FOR (e:Entity)
    REQUIRE e.entity_id IS UNIQUE
    """)


def reset_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def load_documents(tx, rows):
    tx.run("""
    UNWIND $rows AS row
    MERGE (d:Document {document_id: row.document_id})
    SET d += row
    """, rows=rows)


def load_chunks(tx, rows):
    tx.run("""
    UNWIND $rows AS row
    MERGE (c:Chunk {chunk_id: row.chunk_id})
    SET c += row
    WITH c, row
    MATCH (d:Document {document_id: row.document_id})
    MERGE (d)-[:CONTAINS]->(c)
    """, rows=rows)


def load_entities(tx, rows):
    tx.run("""
    UNWIND $rows AS row
    MERGE (e:Entity {entity_id: row.entity_id})
    SET e += row
    """, rows=rows)


def load_mentions(tx, rows):
    tx.run("""
    UNWIND $rows AS row
    MATCH (c:Chunk {chunk_id: row.chunk_id})
    MATCH (e:Entity {entity_id: row.entity_id})
    MERGE (c)-[m:MENTIONS {mention_id: row.mention_id}]->(e)
    SET m += row
    """, rows=rows)


def load_relation_batch(tx, relation_type, rows):
    query = f"""
    UNWIND $rows AS row
    MATCH (src:Entity {{entity_id: row.source_entity_id}})
    MATCH (dst:Entity {{entity_id: row.target_entity_id}})
    MERGE (src)-[r:{relation_type} {{relation_id: row.relation_id}}]->(dst)
    SET r += row
    """
    tx.run(query, rows=rows)


def chunks(iterable, size=500):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_args():
    parser = argparse.ArgumentParser(description="Load processed JSONL graph data into Neo4j.")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with processed JSONL files.")
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per write transaction.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip processed JSONL validation before loading.")
    parser.add_argument("--reset-db", action="store_true", help="Delete all Neo4j nodes and relationships before loading.")
    return parser.parse_args()


def load_file(session, path, batch_size, loader):
    loaded = 0
    for batch in chunks(read_jsonl(path), batch_size):
        session.execute_write(loader, batch)
        loaded += len(batch)
    return loaded


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    batch_size = args.batch_size
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    if not args.skip_validation:
        run_validator(data_dir)
    else:
        print("Skipping processed data validation.")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    summary = {
        "documents": 0,
        "chunks": 0,
        "entities": 0,
        "mentions": 0,
        "relations": 0,
        "relation_types": {},
    }

    with driver.session() as session:
        if args.reset_db:
            print("WARNING: --reset-db was passed. Deleting all nodes and relationships from Neo4j.", flush=True)
            session.execute_write(reset_database)

        session.execute_write(create_constraints)

        summary["documents"] = load_file(
            session,
            data_dir / "documents.jsonl",
            batch_size,
            load_documents,
        )

        summary["chunks"] = load_file(
            session,
            data_dir / "chunks.jsonl",
            batch_size,
            load_chunks,
        )

        summary["entities"] = load_file(
            session,
            data_dir / "entities.jsonl",
            batch_size,
            load_entities,
        )

        summary["mentions"] = load_file(
            session,
            data_dir / "entity_mentions.jsonl",
            batch_size,
            load_mentions,
        )

        relation_groups = {}
        for row in read_jsonl(data_dir / "relations.jsonl"):
            relation_type = safe_relation_type(row.get("relation_type", "RELATED_TO"))
            relation_groups.setdefault(relation_type, []).append(row)

        for relation_type, rows in relation_groups.items():
            summary["relation_types"][relation_type] = len(rows)
            for batch in chunks(rows, batch_size):
                session.execute_write(load_relation_batch, relation_type, batch)
                summary["relations"] += len(batch)

    driver.close()
    print("Graph loaded successfully")
    print("Load summary:")
    print(f"  documents loaded: {summary['documents']}")
    print(f"  chunks loaded: {summary['chunks']}")
    print(f"  entities loaded: {summary['entities']}")
    print(f"  mentions loaded: {summary['mentions']}")
    print(f"  relations loaded: {summary['relations']}")
    print("  relation type distribution:")
    for relation_type, count in sorted(summary["relation_types"].items()):
        print(f"    {relation_type}: {count}")


if __name__ == "__main__":
    main()
