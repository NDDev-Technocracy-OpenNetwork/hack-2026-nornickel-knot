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
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))
from graph_store import find_nodes, load_graph, neighborhood  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
# несколько бесплатных моделей подряд -- если одна упирается в upstream
# rate-limit (бывает часто на :free тарифе), пробуем следующую
MODEL_FALLBACKS = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

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


def get_client() -> OpenAI:
    load_dotenv(ROOT / ".env")
    return OpenAI(api_key=os.environ[API_KEY_ENV], base_url=BASE_URL)


def answer_question(g, client: OpenAI, question: str, hops: int = 2) -> dict:
    """Возвращает {seeds, subgraph_nodes, subgraph_edges, sources, answer}
    либо {seeds: [], answer: None, reason: "..."} если графу нечего сказать.
    Используется и CLI (main), и веб-приложением."""
    seeds = retrieve_seed_nodes(g, question)
    if not seeds:
        return {"seeds": [], "answer": None, "reason": "Ничего не найдено в графе по ключевым словам вопроса."}

    sub = neighborhood(g, seeds, hops=hops)
    facts_text, sources = facts_to_text(g, sub)
    if not facts_text:
        return {"seeds": [g.nodes[s]["canonical_name"] for s in seeds], "answer": None,
                "reason": "В окрестности найденных узлов нет связей (изолированные упоминания)."}

    resp = None
    last_err = None
    used_model = None
    for model in MODEL_FALLBACKS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": ANSWER_PROMPT.format(question=question, facts=facts_text[:15000])}],
                temperature=0.2,
            )
            used_model = model
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if resp is None:
        return {"seeds": [g.nodes[s]["canonical_name"] for s in seeds], "answer": None,
                "reason": f"LLM недоступен: {str(last_err)[:200]}"}
    return {
        "seeds": [g.nodes[s]["canonical_name"] for s in seeds],
        "subgraph_nodes": sub.number_of_nodes(),
        "subgraph_edges": sub.number_of_edges(),
        "sources": sorted(sources),
        "answer": resp.choices[0].message.content,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print('Использование: uv run python3 src/query/ask.py "твой вопрос"')
        sys.exit(1)
    question = sys.argv[1]

    client = get_client()
    g = load_graph()
    result = answer_question(g, client, question)

    print(f"Найдено узлов-затравок: {len(result['seeds'])} -> {result['seeds'][:10]}")
    if result["answer"] is None:
        print(result["reason"])
        return
    print(f"Подграф: {result['subgraph_nodes']} узлов, {result['subgraph_edges']} рёбер, "
          f"{len(result['sources'])} источников\n")
    print("=== ОТВЕТ ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
