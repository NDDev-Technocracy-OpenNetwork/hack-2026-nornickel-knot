"""Прогоняет каждый документ из data/parsed/ через обученный Harmonic целиком,
сохраняет document-эмбеддинги (h1+h2+h3 финальное состояние) в
data/harmonic_runs/doc_embeddings.pt.
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harmonic_embed import ROOT, embed_text, load_model  # noqa: E402

SOURCE_ROOT = ROOT / "data" / "parsed"
OUT_PATH = ROOT / "data" / "harmonic_runs" / "doc_embeddings.pt"


def main() -> None:
    model = load_model()
    entries = []
    t0 = time.perf_counter()

    parsed_files = sorted(SOURCE_ROOT.rglob("*.json"))
    for i, path in enumerate(parsed_files):
        doc = json.loads(path.read_text(encoding="utf-8"))
        text = "\n\n".join(p["text"] for p in doc["pages"])
        vec = embed_text(text, model=model)
        entries.append({"filename": doc["filename"], "category": doc["category"], "vector": vec})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(parsed_files)}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    torch.save(entries, OUT_PATH)
    print(f"\nГотово: {len(entries)} документов, {time.perf_counter() - t0:.1f}s -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
