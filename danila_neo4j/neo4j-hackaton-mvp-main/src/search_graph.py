import argparse
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def parse_args():
    parser = argparse.ArgumentParser(description="Search the Neo4j knowledge graph.")
    parser.add_argument("query", help="Entity search text.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of entities to return.")
    parser.add_argument("--relations-limit", type=int, default=20, help="Maximum semantic relations per entity.")
    parser.add_argument("--mentions-limit", type=int, default=20, help="Maximum source mentions per entity.")
    return parser.parse_args()


def require_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")


def fetch_entities(session, search_text, limit):
    result = session.run(
        """
        MATCH (e:Entity)
        RETURN
          e.entity_id AS entity_id,
          e.name AS name,
          e.normalized_name AS normalized_name,
          e.entity_type AS entity_type,
          e.aliases AS aliases
        ORDER BY entity_type, name, entity_id
        """
    )
    query = search_text.lower()
    matches = []
    for record in result:
        entity = dict(record)
        if entity_matches_query(entity, query):
            entity.pop("aliases", None)
            matches.append(entity)
            if len(matches) >= limit:
                break
    return matches


def fetch_mentions(session, entity_id, limit):
    result = session.run(
        """
        MATCH (e:Entity {entity_id: $entity_id})
        MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[m:MENTIONS]->(e)
        RETURN
          d.title AS document_title,
          d.document_id AS document_id,
          c.chunk_id AS chunk_id,
          coalesce(m.page_start, c.page_start) AS page_start,
          coalesce(m.page_end, c.page_end) AS page_end,
          m.surface_text AS surface_text
        ORDER BY document_title, chunk_id, surface_text
        LIMIT $limit
        """,
        entity_id=entity_id,
        limit=limit,
    )
    return [dict(record) for record in result]


def entity_matches_query(entity, query):
    values = [
        entity.get("name"),
        entity.get("normalized_name"),
    ]
    aliases = entity.get("aliases")
    if isinstance(aliases, list):
        values.extend(aliases)
    elif isinstance(aliases, str):
        values.append(aliases)

    return any(
        query in str(value).lower()
        for value in values
        if value is not None
    )


def fetch_relations(session, entity_id, limit):
    result = session.run(
        """
        MATCH (entity:Entity {entity_id: $entity_id})-[r]-(other:Entity)
        WHERE type(r) <> "MENTIONS"
        WITH r, startNode(r) AS source, endNode(r) AS target
        RETURN
          source.name AS source_name,
          source.entity_type AS source_type,
          type(r) AS relation_type,
          target.name AS target_name,
          target.entity_type AS target_type,
          r.evidence_text AS evidence_text,
          r.confidence AS confidence,
          r.document_id AS document_id,
          r.chunk_id AS chunk_id
        ORDER BY relation_type, source_name, target_name
        LIMIT $limit
        """,
        entity_id=entity_id,
        limit=limit,
    )
    return [dict(record) for record in result]


def print_found_entities(entities):
    print("FOUND ENTITIES")
    for index, entity in enumerate(entities, start=1):
        print(f"{index}. entity_id: {value(entity.get('entity_id'))}")
        print(f"   name: {value(entity.get('name'))}")
        print(f"   normalized_name: {value(entity.get('normalized_name'))}")
        print(f"   entity_type: {value(entity.get('entity_type'))}")
    print()


def print_mentions(entity, mentions):
    print(f"MENTIONS for {entity_label(entity)}")
    if not mentions:
        print("  No source mentions found.")
        print()
        return

    for index, mention in enumerate(mentions, start=1):
        page_range = format_page_range(mention.get("page_start"), mention.get("page_end"))
        print(f"{index}. document title: {value(mention.get('document_title'))}")
        print(f"   document_id: {value(mention.get('document_id'))}")
        print(f"   chunk_id: {value(mention.get('chunk_id'))}")
        print(f"   page_start/page_end: {page_range}")
        print(f"   surface_text: {value(mention.get('surface_text'))}")
    print()


def print_relations(entity, relations):
    print(f"RELATIONS for {entity_label(entity)}")
    if not relations:
        print("  No semantic Entity -> Entity relations found.")
        print()
        return

    for index, relation in enumerate(relations, start=1):
        print(
            f"{index}. {value(relation.get('source_name'))}"
            f" ({value(relation.get('source_type'))})"
            f" -[{value(relation.get('relation_type'))}]->"
            f" {value(relation.get('target_name'))}"
            f" ({value(relation.get('target_type'))})"
        )
        print(f"   evidence_text: {value(relation.get('evidence_text'))}")
        print(f"   confidence: {value(relation.get('confidence'))}")
        print(f"   document_id: {value(relation.get('document_id'))}")
        print(f"   chunk_id: {value(relation.get('chunk_id'))}")
    print()


def entity_label(entity):
    return f"{value(entity.get('name'))} [{value(entity.get('entity_id'))}]"


def format_page_range(page_start, page_end):
    return f"{value(page_start)}/{value(page_end)}"


def value(raw):
    if raw is None or raw == "":
        return "-"
    return str(raw)


def main():
    args = parse_args()
    require_positive(args.limit, "--limit")
    require_positive(args.relations_limit, "--relations-limit")
    require_positive(args.mentions_limit, "--mentions-limit")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()
        with driver.session() as session:
            entities = fetch_entities(session, args.query, args.limit)
            if not entities:
                print(f'No entities found for query: "{args.query}"')
                print("Try inspecting existing entities with:")
                print("MATCH (e:Entity) RETURN e.name, e.entity_type LIMIT 20;")
                return 0

            print(f'QUERY: "{args.query}"')
            print()
            print_found_entities(entities)

            for entity in entities:
                mentions = fetch_mentions(session, entity["entity_id"], args.mentions_limit)
                relations = fetch_relations(session, entity["entity_id"], args.relations_limit)
                print_mentions(entity, mentions)
                print_relations(entity, relations)
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
