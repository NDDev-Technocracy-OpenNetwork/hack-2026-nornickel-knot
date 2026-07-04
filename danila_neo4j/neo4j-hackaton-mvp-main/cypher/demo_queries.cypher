// Neo4j demo queries for the processed JSONL knowledge graph.
// Run queries one block at a time in Neo4j Browser or cypher-shell.

// ---------------------------------------------------------------------------
// 1. Counts
// ---------------------------------------------------------------------------

// Count Document nodes.
MATCH (d:Document)
RETURN count(d) AS documents_count;

// Count Chunk nodes.
MATCH (c:Chunk)
RETURN count(c) AS chunks_count;

// Count Entity nodes.
MATCH (e:Entity)
RETURN count(e) AS entities_count;

// Count all relationships by type.
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(r) AS count
ORDER BY relationship_type;

// Count source-tracing and semantic relationship groups separately.
MATCH (:Document)-[contains:CONTAINS]->(:Chunk)
WITH count(contains) AS document_contains_chunk
MATCH (:Chunk)-[mentions:MENTIONS]->(:Entity)
WITH document_contains_chunk, count(mentions) AS chunk_mentions_entity
MATCH (:Entity)-[semantic]->(:Entity)
RETURN
  document_contains_chunk,
  chunk_mentions_entity,
  count(semantic) AS entity_semantic_relations;


// ---------------------------------------------------------------------------
// 2. Source tracing
// ---------------------------------------------------------------------------

// Document -> Chunk: inspect chunks belonging to a document.
MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
RETURN
  d.document_id AS document_id,
  d.title AS document_title,
  c.chunk_id AS chunk_id,
  c.chunk_index AS chunk_index,
  c.page_start AS page_start,
  c.page_end AS page_end,
  left(c.text, 300) AS chunk_preview
ORDER BY document_id, chunk_index
LIMIT 25;

// Document -> Chunk -> Entity: trace extracted entities back to source chunks.
MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[m:MENTIONS]->(e:Entity)
RETURN
  d.title AS document_title,
  c.chunk_id AS chunk_id,
  coalesce(m.page_start, c.page_start) AS page_start,
  coalesce(m.page_end, c.page_end) AS page_end,
  m.surface_text AS surface_text,
  e.entity_id AS entity_id,
  e.name AS entity_name,
  e.entity_type AS entity_type,
  m.confidence AS confidence
ORDER BY document_title, chunk_id, surface_text
LIMIT 50;

// Mentions with document title, chunk id, page range, and surface text.
MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[m:MENTIONS]->(e:Entity)
RETURN
  d.title AS document_title,
  c.chunk_id AS chunk_id,
  coalesce(m.page_start, c.page_start) AS page_start,
  coalesce(m.page_end, c.page_end) AS page_end,
  m.surface_text AS surface_text,
  e.name AS entity_name
ORDER BY document_title, chunk_id
LIMIT 50;

// Source snippets for a selected entity substring.
WITH "никель" AS query
MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[m:MENTIONS]->(e:Entity)
WHERE toLower(coalesce(e.name, "")) CONTAINS toLower(query)
   OR toLower(coalesce(e.normalized_name, "")) CONTAINS toLower(query)
RETURN
  e.name AS entity_name,
  d.title AS document_title,
  c.chunk_id AS chunk_id,
  c.page_start AS page_start,
  c.page_end AS page_end,
  m.surface_text AS surface_text,
  left(c.text, 500) AS chunk_preview
ORDER BY entity_name, document_title, chunk_id
LIMIT 20;


// ---------------------------------------------------------------------------
// 3. Semantic relations
// ---------------------------------------------------------------------------

// All Entity -> Entity semantic relations except reserved MENTIONS.
MATCH (source:Entity)-[r]->(target:Entity)
WHERE type(r) <> "MENTIONS"
RETURN
  source.name AS source_name,
  source.entity_type AS source_type,
  type(r) AS relation_type,
  target.name AS target_name,
  target.entity_type AS target_type,
  r.confidence AS confidence
ORDER BY relation_type, source_name, target_name
LIMIT 100;

// Useful R&D relation types.
MATCH (source:Entity)-[r]->(target:Entity)
WHERE type(r) IN [
  "AFFECTS",
  "IMPROVES",
  "WORSENS",
  "MEASURES",
  "PRODUCES",
  "USES",
  "DEPENDS_ON"
]
RETURN
  source.name AS source_name,
  type(r) AS relation_type,
  target.name AS target_name,
  r.evidence_text AS evidence_text,
  r.confidence AS confidence
ORDER BY relation_type, confidence DESC
LIMIT 100;

// Relations with evidence_text and confidence, traced back to source chunks.
MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
MATCH (source:Entity)-[r]->(target:Entity)
WHERE r.chunk_id = c.chunk_id
  AND type(r) <> "MENTIONS"
  AND r.evidence_text IS NOT NULL
  AND trim(r.evidence_text) <> ""
RETURN
  type(r) AS relation_type,
  source.name AS source_name,
  target.name AS target_name,
  r.evidence_text AS evidence_text,
  r.confidence AS confidence,
  d.title AS document_title,
  c.chunk_id AS chunk_id,
  c.page_start AS page_start,
  c.page_end AS page_end
ORDER BY confidence DESC, relation_type
LIMIT 100;


// ---------------------------------------------------------------------------
// 4. Data quality in Neo4j
// ---------------------------------------------------------------------------

// Chunks without a source Document.
MATCH (c:Chunk)
WHERE NOT (:Document)-[:CONTAINS]->(c)
RETURN c.chunk_id AS chunk_id, c.document_id AS expected_document_id
ORDER BY chunk_id
LIMIT 100;

// Entities without any relationship.
MATCH (e:Entity)
WHERE NOT (e)--()
RETURN
  e.entity_id AS entity_id,
  e.name AS entity_name,
  e.entity_type AS entity_type
ORDER BY entity_type, entity_name
LIMIT 100;

// Semantic Entity -> Entity relations without evidence_text.
MATCH (source:Entity)-[r]->(target:Entity)
WHERE type(r) <> "MENTIONS"
  AND (r.evidence_text IS NULL OR trim(r.evidence_text) = "")
RETURN
  source.name AS source_name,
  type(r) AS relation_type,
  target.name AS target_name,
  r.relation_id AS relation_id
ORDER BY relation_type, relation_id
LIMIT 100;

// Chunks with empty text.
MATCH (c:Chunk)
WHERE c.text IS NULL OR trim(c.text) = ""
RETURN
  c.chunk_id AS chunk_id,
  c.document_id AS document_id,
  c.chunk_index AS chunk_index
ORDER BY document_id, chunk_index
LIMIT 100;

// Entity -> MENTIONS -> Entity should return zero records.
// MENTIONS is reserved for (:Chunk)-[:MENTIONS]->(:Entity).
MATCH (source:Entity)-[r:MENTIONS]->(target:Entity)
RETURN
  source.entity_id AS source_entity_id,
  source.name AS source_name,
  target.entity_id AS target_entity_id,
  target.name AS target_name,
  r.relation_id AS relation_id
LIMIT 100;


// ---------------------------------------------------------------------------
// 5. Search examples
// ---------------------------------------------------------------------------

// Search entity by substring in name or normalized_name.
WITH "никель" AS query
MATCH (e:Entity)
WHERE toLower(coalesce(e.name, "")) CONTAINS toLower(query)
   OR toLower(coalesce(e.normalized_name, "")) CONTAINS toLower(query)
RETURN
  e.entity_id AS entity_id,
  e.name AS entity_name,
  e.normalized_name AS normalized_name,
  e.entity_type AS entity_type
ORDER BY entity_type, entity_name
LIMIT 25;

// Show semantic neighborhood around found entities.
WITH "никель" AS query
MATCH (center:Entity)
WHERE toLower(coalesce(center.name, "")) CONTAINS toLower(query)
   OR toLower(coalesce(center.normalized_name, "")) CONTAINS toLower(query)
MATCH path = (center)-[r]-(neighbor:Entity)
WHERE type(r) <> "MENTIONS"
RETURN path
LIMIT 50;

// Show documents and chunks where found entities are mentioned.
WITH "никель" AS query
MATCH (e:Entity)
WHERE toLower(coalesce(e.name, "")) CONTAINS toLower(query)
   OR toLower(coalesce(e.normalized_name, "")) CONTAINS toLower(query)
MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[m:MENTIONS]->(e)
RETURN
  e.name AS entity_name,
  d.title AS document_title,
  c.chunk_id AS chunk_id,
  c.page_start AS page_start,
  c.page_end AS page_end,
  m.surface_text AS surface_text,
  left(c.text, 500) AS chunk_preview
ORDER BY entity_name, document_title, chunk_id
LIMIT 50;
