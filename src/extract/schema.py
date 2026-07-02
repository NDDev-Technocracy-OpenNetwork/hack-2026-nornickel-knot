"""Ontology for fact extraction, taken directly from the hackathon task spec
(info.txt): entity types, relation types, and the JSON shape the LLM must return."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

EntityType = Literal[
    "Material", "Process", "Equipment", "Property",
    "Experiment", "Publication", "Expert", "Facility",
]

RelationType = Literal[
    "uses_material", "operates_at_condition", "produces_output",
    "described_in", "validated_by", "contradicts",
]


class Entity(BaseModel):
    id: str = Field(description="short local id, unique within this chunk, e.g. 'e1'")
    type: EntityType
    name: str = Field(description="entity name as it appears in the source text")
    canonical_name: str = Field(
        description="normalized name for matching synonyms/translations across documents, "
        "e.g. 'электроэкстракция' and 'electrowinning' both -> 'electrowinning'"
    )


class Relation(BaseModel):
    source_id: str = Field(description="id of an entity defined in this chunk's entities list")
    relation: RelationType
    target_id: str = Field(description="id of an entity defined in this chunk's entities list")
    numeric_value: Optional[float] = Field(default=None, description="numeric value if the relation carries one")
    unit: Optional[str] = Field(default=None, description="unit for numeric_value, e.g. 'mg/l', 'm3/h', 'C'")
    condition: Optional[str] = Field(default=None, description="qualifying condition, e.g. 'холодный климат'")
    confidence: float = Field(description="model's own confidence that this relation is explicitly supported by the text, 0-1")


class ExtractionResult(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
