from src.models import MaterialFact
import json


def load_material_facts(path: str) -> list[MaterialFact]:
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    return [MaterialFact(**item) for item in raw_data]


if __name__ == "__main__":
    facts = load_material_facts("data/extracted/material_facts.json")
    print(facts)
