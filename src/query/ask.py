"""CLI: question -> naive keyword match against graph nodes -> neighborhood
subgraph -> LLM synthesizes an answer citing source documents.

MVP placeholder note: node retrieval here is plain substring matching over
entity names. This is the slot Harmonic is meant to fill later (full-corpus,
whole-document semantic filtering) -- swapping it in later should not require
touching anything past retrieve_seed_nodes().
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))
from graph_store import find_nodes, load_graph, neighborhood  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gpt-oss-120b"

STOPWORDS = {
    "какие", "какой", "какая", "что", "как", "для", "при", "если", "или",
    "это", "все", "есть", "может", "можно", "по", "на", "в", "и", "с", "из",
    "the", "a", "an", "of", "for", "in", "on", "and", "or", "is", "are", "what",
}


def extract_keywords(question: str) -> list[str]:
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", question.lower())
    return [w for w in words if len(w) > 3 and w not in STOPWORDS]


def retrieve_seed_nodes(g, question: str) -> list[str]:
    seeds: set[str] = set()
    for kw in extract_keywords(question):
        seeds.update(find_nodes(g, kw, limit=5))
    return list(seeds)


def facts_to_text(g, sub) -> tuple[str, set[str]]:
    lines = []
    sources: set[str] = set()
    for u, v, data in sub.edges(data=True):
        u_name = g.nodes[u]["canonical_name"]
        v_name = g.nodes[v]["canonical_name"]
        extra = ""
        if data.get("numeric_value") is not None:
            extra += f", value={data['numeric_value']}{data.get('unit') or ''}"
        if data.get("condition"):
            extra += f", condition={data['condition']}"
        src = data["source_file"]
        sources.add(src)
        lines.append(f"- {u_name} --[{data['relation']}]--> {v_name}{extra} (source: {src}, p.{data['page_start']}-{data['page_end']})")
    return "\n".join(lines), sources


ANSWER_PROMPT = """Ты отвечаешь на вопрос по карте знаний горно-металлургической
R&D-области, используя ТОЛЬКО факты ниже. Если фактов недостаточно для полного
ответа — прямо скажи, чего не хватает, не выдумывай.

Вопрос: {question}

Известные факты (формат: сущность --[связь]--> сущность, источник):
{facts}

Дай краткий структурированный ответ на русском, и в конце отдельным списком
укажи, из каких источников (файлов) он собран.
"""


def main() -> None:
    if len(sys.argv) < 2:
        print('Использование: uv run python3 src/query/ask.py "твой вопрос"')
        sys.exit(1)
    question = sys.argv[1]

    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["CEREBRAS_API_KEY"], base_url="https://api.cerebras.ai/v1")

    g = load_graph()
    seeds = retrieve_seed_nodes(g, question)
    print(f"Найдено узлов-затравок: {len(seeds)} -> {[g.nodes[s]['canonical_name'] for s in seeds][:10]}")

    if not seeds:
        print("Ничего не найдено в графе по ключевым словам вопроса.")
        return

    sub = neighborhood(g, seeds, hops=2)
    facts_text, sources = facts_to_text(g, sub)
    print(f"Подграф: {sub.number_of_nodes()} узлов, {sub.number_of_edges()} рёбер, {len(sources)} источников\n")

    if not facts_text:
        print("В окрестности найденных узлов нет связей (изолированные упоминания).")
        return

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": ANSWER_PROMPT.format(question=question, facts=facts_text[:15000])}],
        temperature=0.2,
    )
    print("=== ОТВЕТ ===")
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
