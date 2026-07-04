import os

from dotenv import load_dotenv
from fastapi import FastAPI
from neo4j import GraphDatabase

from src.graph_reader import GraphReader

load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

driver = GraphDatabase.driver(URI, auth=AUTH)
reader = GraphReader(driver)

app = FastAPI(title="Graph Hackathon API")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/materials-for-kpi")
def get_materials_for_kpi(name: str) -> list[dict]:
    return reader.find_materials_for_kpi(name)


@app.get("/recommendations")
def get_recommendations(name: str) -> dict:
    evidence = reader.find_materials_for_kpi(name)

    if not evidence:
        return {
            "kpi": name,
            "best_candidate": None,
            "alternatives": [],
        }

    best = evidence[0]
    alternatives = evidence[1:]

    return {
        "kpi": name,
        "best_candidate": {
            "material": best["material"],
            "score": best["value"],
            "reason": (
                f"{best['material']} is linked to {best['kpi']} "
                f"through {best['property']} based on "
                f"{best['experiment']} from {best['article']}."
            ),
            "article": best["article"],
            "experiment": best["experiment"],
            "method": best["method"],
            "evidence_path": best["evidence_path"],
        },
        "alternatives": [
            {
                "material": item["material"],
                "score": item["value"],
                "property": item["property"],
                "evidence_path": item["evidence_path"],
            }
            for item in alternatives
        ],
    }
