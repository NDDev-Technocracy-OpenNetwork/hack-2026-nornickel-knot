"""Builds a networkx MultiDiGraph from data/extracted/*.json and saves/loads it.

Nodes are deduplicated by (type, canonical_name) so the same concept mentioned
across different documents (or in Russian vs English) collapses into one node,
with every source document tracked instead of overwritten.
"""

import json
import pickle
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "data" / "extracted"
GRAPH_PATH = ROOT / "data" / "graph" / "knowledge_graph.pkl"


def node_key(entity_type: str, canonical_name: str) -> str:
    return f"{entity_type}:{canonical_name.strip().lower()}"


def build_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    for doc_path in sorted(SOURCE_ROOT.rglob("*.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))

        local_to_key: dict[str, str] = {}
        for e in doc["entities"]:
            key = node_key(e["type"], e["canonical_name"])
            local_to_key[e["id"]] = key
            if g.has_node(key):
                g.nodes[key]["surface_forms"].add(e["name"])
                g.nodes[key]["mentions"] += 1
                g.nodes[key]["sources"].add(e["provenance"]["source_file"])
            else:
                g.add_node(
                    key,
                    type=e["type"],
                    canonical_name=e["canonical_name"],
                    surface_forms={e["name"]},
                    mentions=1,
                    sources={e["provenance"]["source_file"]},
                )

        for r in doc["relations"]:
            src = local_to_key.get(r["source_id"])
            tgt = local_to_key.get(r["target_id"])
            if src is None or tgt is None or not g.has_node(src) or not g.has_node(tgt):
                continue
            g.add_edge(
                src, tgt,
                relation=r["relation"],
                numeric_value=r["numeric_value"],
                unit=r["unit"],
                condition=r["condition"],
                confidence=r["confidence"],
                source_file=r["provenance"]["source_file"],
                category=r["provenance"]["category"],
                page_start=r["provenance"]["page_start"],
                page_end=r["provenance"]["page_end"],
            )

    return g


def save_graph(g: nx.MultiDiGraph) -> None:
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(g, f)


def load_graph() -> nx.MultiDiGraph:
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def find_nodes(g: nx.MultiDiGraph, query: str, limit: int = 15) -> list[str]:
    q = query.strip().lower()
    hits = []
    for key, data in g.nodes(data=True):
        if q in data["canonical_name"].lower() or any(q in s.lower() for s in data["surface_forms"]):
            hits.append(key)
    return hits[:limit]


def neighborhood(g: nx.MultiDiGraph, keys: list[str], hops: int = 2) -> nx.MultiDiGraph:
    seen = set(keys)
    frontier = set(keys)
    undirected = g.to_undirected(as_view=True)
    for _ in range(hops):
        next_frontier = set()
        for k in frontier:
            if k in undirected:
                next_frontier |= set(undirected.neighbors(k))
        next_frontier -= seen
        seen |= next_frontier
        frontier = next_frontier
    return g.subgraph(seen).copy()


def print_summary(g: nx.MultiDiGraph) -> None:
    by_type: dict[str, int] = {}
    for _, data in g.nodes(data=True):
        by_type[data["type"]] = by_type.get(data["type"], 0) + 1
    by_relation: dict[str, int] = {}
    for _, _, data in g.edges(data=True):
        by_relation[data["relation"]] = by_relation.get(data["relation"], 0) + 1

    print(f"Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
    print("By entity type:", dict(sorted(by_type.items(), key=lambda x: -x[1])))
    print("By relation type:", dict(sorted(by_relation.items(), key=lambda x: -x[1])))

    top = sorted(g.nodes(data=True), key=lambda x: -x[1]["mentions"])[:10]
    print("\nTop mentioned entities:")
    for key, data in top:
        print(f"  {data['mentions']:<3} {data['type']:<12} {data['canonical_name']} (sources: {len(data['sources'])})")


def main() -> None:
    g = build_graph()
    save_graph(g)
    print_summary(g)
    print(f"\nSaved to {GRAPH_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
