# Neo4j Demo Queries

This guide explains how to use `cypher/demo_queries.cypher` during a quick R&D knowledge graph demo.

## What to run first

Start with the count queries:

1. Count `Document`, `Chunk`, and `Entity` nodes.
2. Count relationships by type.
3. Count source-tracing relationships separately from `Entity -> Entity` semantic relations.

These queries prove that the loader imported the expected graph shape before moving into examples.

## What to show in the demo

Show the graph in this order:

1. `Document -> Chunk` to prove that source documents are split into traceable chunks.
2. `Document -> Chunk -> Entity` to show extracted entities with document title, chunk id, page range, and `surface_text`.
3. Useful R&D semantic relations such as `AFFECTS`, `IMPROVES`, `MEASURES`, `PRODUCES`, `USES`, and `DEPENDS_ON`.
4. Search examples for an entity substring, then expand the entity neighborhood.

Use the search examples with a domain term such as `никель`, `медь`, `обогащение`, or another known entity substring from the loaded data.

## Source tracing proof

The source tracing proof is the chain:

```cypher
(Document)-[:CONTAINS]->(Chunk)-[:MENTIONS]->(Entity)
```

The strongest demo query is the one returning:

- document title;
- chunk id;
- page start and page end;
- mention `surface_text`;
- entity name and type;
- chunk text preview.

This shows that every displayed entity mention can be traced back to the source document context.

## Entity-to-Entity knowledge graph proof

The semantic knowledge graph proof is the chain:

```cypher
(Entity)-[semantic_relation]->(Entity)
```

Use the queries that:

- list all `Entity -> Entity` relations except reserved `MENTIONS`;
- filter to useful R&D relation types;
- return `evidence_text`, `confidence`, document title, and chunk id.

These queries show that the graph is not only full-text extraction: it contains typed semantic relations between domain entities with evidence and source traceability.

## Why MENTIONS Is Reserved

`MENTIONS` is a source-tracing relationship, not a semantic relationship between entities. It should only represent:

```cypher
(Chunk)-[:MENTIONS]->(Entity)
```

This means "this exact source chunk mentions this entity." Semantic knowledge graph edges should use domain relation types such as `AFFECTS`, `USES`, `PRODUCES`, `MEASURES`, or fallback `RELATED_TO`.

The data quality query `Entity -> MENTIONS -> Entity should return zero records` verifies that no semantic `Entity -> Entity` relation accidentally uses the reserved `MENTIONS` type.

## Data quality checks

Run the data quality section after the demo flow if reviewers ask about reliability. The key checks are:

- chunks without a source document;
- entities without any relationship;
- semantic relations without `evidence_text`;
- chunks with empty text;
- `Entity -> MENTIONS -> Entity`, which should return zero records.

Empty results for structural checks are a good sign. Non-empty evidence checks should be reviewed as data quality issues, not loader failures.
