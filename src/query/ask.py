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

import torch
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harmonic_train"))
from graph_store import find_nodes, load_graph, neighborhood  # noqa: E402
from harmonic_embed import CHECKPOINT_PATH, embed_text, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOC_EMBEDDINGS_PATH = ROOT / "data" / "harmonic_runs" / "doc_embeddings.pt"

# Yandex первым -- жёсткое требование задания: модель должна быть доступна
# из РФ без обхода ограничений. Cerebras (проверено -- 403 без прокси) и
# OpenRouter остаются в цепочке как запасные, но не для официальной сдачи.
# gpt-oss-120b на Yandex -- reasoning-модель, без большого max_tokens отдаёт
# пустой content (бюджет уходит на скрытые раздумья) -- поэтому primary
# именно yandexgpt/latest, не reasoning, отвечает сразу и предсказуемо.
PROVIDERS = [
    # Yandex первым -- жёсткое условие хакатона: API с доступом из РФ.
    # Cerebras/OpenRouter в хвосте только как аварийные фоллбэки (Cerebras
    # из РФ доступен только через прокси -- не для официальной сдачи).
    ("yandex", "https://llm.api.cloud.yandex.net/v1", "YANDEX_API_KEY", "gpt://__FOLDER__/yandexgpt/latest"),
    ("yandex", "https://llm.api.cloud.yandex.net/v1", "YANDEX_API_KEY", "gpt://__FOLDER__/gpt-oss-120b/latest"),
    ("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "gpt-oss-120b"),
    ("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-oss-120b:free"),
    ("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    ("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "qwen/qwen3-next-80b-a3b-instruct:free"),
    ("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "nvidia/nemotron-3-super-120b-a12b:free"),
]
# оставлено для обратной совместимости с CLI-флагом retrieval, не используется
# напрямую в answer_question -- см. PROVIDERS
BASE_URL = PROVIDERS[0][1]
API_KEY_ENV = PROVIDERS[0][2]

STOPWORDS = {
    "какие", "какой", "какая", "что", "как", "для", "при", "если", "или",
    "это", "все", "есть", "может", "можно", "по", "на", "в", "и", "с", "из",
    "the", "a", "an", "of", "for", "in", "on", "and", "or", "is", "are", "what",
    # мета-слова из типовой формулировки вопроса ("литературный обзор...
    # отечественная и мировая практика") -- рамка задания, не тема. Найдено
    # на реальном вопросе жюри: эти слова редки как начало имени сущности
    # в графе (просто не техническая лексика), из-за чего наивный "редкий
    # префикс = специфичный" ранжировщик ошибочно ставил их выше настоящей
    # темы вопроса ("очистка", "шахтных").
    "обзор", "литературный", "литературу", "выполнить", "провести", "анализ",
    "практика", "практики", "мировая", "мировой", "мировом", "отечественная",
    "отечественный", "отечественную", "существующих", "техническая",
    "технической", "технических", "решений", "решения", "источников",
}


def extract_keywords(question: str) -> list[str]:
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", question.lower())
    return [w for w in words if len(w) > 3 and w not in STOPWORDS]


def retrieve_seed_nodes(g, question: str) -> list[str]:
    """Baseline: наивное совпадение слов вопроса с именами узлов графа."""
    seeds: set[str] = set()
    for kw in extract_keywords(question):
        seeds.update(find_nodes(g, kw, limit=5))
    return list(seeds)


_doc_embeddings = None


def load_doc_embeddings():
    global _doc_embeddings
    if _doc_embeddings is None:
        _doc_embeddings = torch.load(DOC_EMBEDDINGS_PATH, weights_only=False)
    return _doc_embeddings


def retrieve_seed_nodes_harmonic(g, question: str, top_k: int = 4) -> tuple[list[str], list[tuple[str, float]]]:
    """Harmonic (эмбеддинг всего документа) + точное совпадение по именам
    сущностей -- гибрид. Проблема чистого эмбеддинга: документ-"сборник"
    (журнальный номер с десятком разных статей) даёт усреднённый вектор,
    который может не попасть в top_k даже если внутри есть точный ответ --
    подтверждено на реальном вопросе жюри про штейн/шлак: документ с фактами
    "штейн"/"шлак" был вне top-4 по эмбеддингу, хотя термины буквально есть
    в тексте. Точное совпадение имени сущности с словом вопроса -- сигнал
    не хуже похожести векторов, и не разбавляется остальным содержимым
    документа. Добавляем такие документы в seed-набор, не заменяя эмбеддинг
    (объединение, не замена -- эмбеддинг всё ещё ловит смысловые совпадения
    без точных слов)."""
    entries = load_doc_embeddings()
    model = load_model()
    q_vec = embed_text(question, model=model)

    scored = []
    for e in entries:
        sim = torch.nn.functional.cosine_similarity(q_vec, e["vector"], dim=0).item()
        scored.append((e["filename"], sim))
    scored.sort(key=lambda x: -x[1])
    top_files = {f for f, _ in scored[:top_k]}

    kw_nodes: list[str] = []
    keywords = set(extract_keywords(question))
    if keywords:
        # сравнение по префиксу (первые 4 символа), не по точному слову --
        # русские падежные окончания ("штейном" в вопросе vs "штейн" в
        # графе) не совпадут как строки, но совпадут по стеблю/префиксу
        kw_prefixes = {w[:4] for w in keywords if len(w) >= 4}

        # найдено на реальном вопросе жюри: частые префиксы вроде "мето"
        # (от "методов" -- совпадает с сотнями узлов "метод X...") или
        # "горн" (от "горно" -- совпадает с "горные породы") набирают тысячи
        # совпадений и вытесняют редкие/специфичные ("очис", "шахт") из
        # первых max_nodes при обрезке в neighborhood(). Считаем частоту
        # каждого префикса СНАЧАЛА, затем сортируем найденные узлы так,
        # чтобы редкий (специфичный) префикс шёл первым -- редкость сигнала
        # важнее порядка вставки в граф.
        matches: list[tuple[str, str]] = []  # (matched_prefix, node_key)
        for key, data in g.nodes(data=True):
            name_words = re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", data["canonical_name"].lower())
            node_prefixes = {w[:4] for w in name_words if len(w) >= 4} & kw_prefixes
            if node_prefixes:
                # если узел совпал по нескольким префиксам -- берём самый
                # редкий из них как представителя для сортировки
                matches.append((key, node_prefixes))

        prefix_freq: dict[str, int] = {}
        for _, prefixes in matches:
            for p in prefixes:
                prefix_freq[p] = prefix_freq.get(p, 0) + 1

        def _specificity(item: tuple[str, set]) -> int:
            _, prefixes = item
            return min(prefix_freq[p] for p in prefixes)  # реже -- специфичнее

        matches.sort(key=_specificity)
        kw_nodes = [key for key, _ in matches]

    # kw_nodes -- первыми: neighborhood() обрезает seed-список до max_nodes,
    # и с общими словами вопроса (после/между/покажите тоже проходят фильтр
    # длины) набирается несколько тысяч совпадений -- если положить их
    # вперемешку с embedding-узлами, точное совпадение по нужному термину
    # может просто не попасть в обрезанные первые max_nodes.
    seen_keys = set(kw_nodes)
    other_nodes = [
        key for key, data in g.nodes(data=True)
        if key not in seen_keys and data["sources"] & top_files
    ]
    seeds = kw_nodes + other_nodes
    return seeds, scored[:top_k]


def _harmonic_missing_reason() -> str | None:
    if not DOC_EMBEDDINGS_PATH.exists():
        return f"нет {DOC_EMBEDDINGS_PATH.relative_to(ROOT)}"
    if not CHECKPOINT_PATH.exists():
        return f"нет {CHECKPOINT_PATH.relative_to(ROOT)}"
    return None


def retrieve_seed_nodes_auto(
    g,
    question: str,
    preferred: str = "harmonic",
    top_k: int = 4,
) -> tuple[list[str], list[tuple[str, float]] | None, str, str | None]:
    """Retrieval with a safe fallback for demo environments.

    The repository can contain precomputed document embeddings without the
    trained Harmonic checkpoint. In that case the web app must still show the
    graph instead of returning HTTP 500.
    """
    if preferred == "harmonic":
        missing = _harmonic_missing_reason()
        if missing:
            return (
                retrieve_seed_nodes(g, question),
                None,
                "keyword",
                f"Harmonic недоступен: {missing}. Использован keyword-ретрив.",
            )
        try:
            seeds, doc_scores = retrieve_seed_nodes_harmonic(g, question, top_k=top_k)
            return seeds, doc_scores, "harmonic", None
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            return (
                retrieve_seed_nodes(g, question),
                None,
                "keyword",
                f"Harmonic недоступен: {str(exc)[:180]}. Использован keyword-ретрив.",
            )

    return retrieve_seed_nodes(g, question), None, "keyword", None


def subgraph_to_viz(g, sub, max_nodes: int = 36) -> dict:
    """Компактное представление подграфа для отрисовки в браузере (a-la
    Obsidian graph view): top-N узлов по степени внутри подграфа + рёбра
    между ними."""
    degrees = sorted(sub.degree(), key=lambda x: -x[1])
    keep = {k for k, _ in degrees[:max_nodes]}
    nodes = [
        {"id": k, "label": g.nodes[k]["canonical_name"], "type": g.nodes[k]["type"],
         "degree": d}
        for k, d in degrees[:max_nodes]
    ]
    edges = []
    seen = set()
    for u, v, data in sub.edges(data=True):
        if u not in keep or v not in keep or u == v:
            continue
        pair = (u, v) if u < v else (v, u)
        if pair in seen:
            continue
        seen.add(pair)
        edges.append({"source": u, "target": v, "relation": data["relation"]})
    return {"nodes": nodes, "edges": edges}


def facts_to_text(g, sub) -> tuple[str, set[str]]:
    """Пробовал сортировать факты по рангу документа-источника перед обрезкой --
    оказалось контрпродуктивно: часть нужных фактов находится в подграфе
    именно через 2-хоп связь из документов, которые НЕ входят в top-k по
    сходству (ранг "неизвестно") -- сортировка топила ровно их. Настоящая
    причина обрезки в другом: наши fallback-модели держат 128K-1M токенов
    контекста, а лимит был 15-28 тысяч СИМВОЛОВ -- на два порядка меньше
    доступного. Правильный фикс -- не резать, а не резать так рано."""
    lines = []
    sources: set[str] = set()
    connected: set[str] = set()
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
        connected.add(u)
        connected.add(v)
        lines.append(f"- {u_name} --[{data['relation']}]--> {v_name}{extra} (source: {src}, p.{data['page_start']}-{data['page_end']})")

    # Узлы без связей (LLM выделила сущность, но не извлекла отношение к
    # ней) иначе пропадают полностью из текста для ответа -- даже если
    # ретрив правильно нашёл документ и термин, при facts_to_text по
    # рёбрам он теряется. Даём модели хотя бы упоминание "сущность
    # встречается в источнике X" вместо полного молчания.
    for key, data in sub.nodes(data=True):
        if key in connected:
            continue
        for src in data["sources"]:
            sources.add(src)
        src_list = ", ".join(sorted(data["sources"])[:3])
        lines.append(f"- упоминается: {data['canonical_name']} ({data['type']}) (source: {src_list})")

    return "\n".join(lines), sources


def facts_preview(facts_text: str, max_lines: int = 14, max_chars: int = 6000) -> str:
    lines = [line for line in facts_text.splitlines() if line.strip()]
    if not lines:
        return ""
    preview_lines = lines[:max_lines]
    preview = "\n".join(preview_lines)
    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip() + "..."
    rest = len(lines) - len(preview_lines)
    if rest > 0:
        preview += f"\n- ... ещё {rest} фактов в найденном подграфе"
    return preview


ANSWER_PROMPT = """Ты отвечаешь на вопрос по карте знаний горно-металлургической
R&D-области, используя ТОЛЬКО факты ниже. Не выдумывай факты и цифры.

Вопрос: {question}

Известные факты (формат: сущность --[связь]--> сущность, источник):
{facts}

Правила ответа:
- Если ниже есть факты, релевантные вопросу (даже частично, даже по смежной
  теме) -- начни сразу с них. НЕ открывай ответ фразой "нет информации" или
  "данных недостаточно", если ты в следующем же предложении приводишь
  релевантные факты -- это противоречит само себе и читается как ошибка.

  Пример НЕПРАВИЛЬНОГО ответа (так делать нельзя):
  "В данных нет информации о методах обработки шахтных вод. Однако
  упоминается технология очистки шахтной воды с извлечением металлов."
  -- это самопротиворечие: сначала отрицание, затем опровержение своего же
  отрицания в той же фразе.

  Пример ПРАВИЛЬНОГО ответа для той же ситуации:
  "По теме обработки шахтных вод найдена технология комплексной очистки
  с попутным извлечением цветных металлов [источник]. Данных о других
  методах (например, реагентной очистке) в базе нет."
  -- сразу факт, потом (если нужно) в чём именно пробел.
- Если факты покрывают вопрос лишь частично -- изложи, что есть, и ТОЛЬКО
  в конце одной фразой отметь, чего именно не хватает для полного ответа.
- Если релевантных фактов нет вообще -- тогда прямо и коротко скажи, что
  по этому вопросу в базе ничего нет, без выдумывания.

Дай краткий структурированный ответ на русском, и в конце отдельным списком
укажи, из каких источников (файлов) он собран.
"""


_clients: dict[str, OpenAI] = {}


def get_client() -> None:
    """Оставлено для обратной совместимости с CLI/веб-приложением, которые
    раньше строили клиент заранее и передавали в answer_question. Теперь
    answer_question сам перебирает провайдеров (Cerebras, затем OpenRouter
    fallback-цепочка) и строит клиентов лениво -- эта функция просто грузит
    .env, чтобы ключи были доступны."""
    load_dotenv(ROOT / ".env")


def _client_for(base_url: str, key_env: str) -> OpenAI:
    if base_url not in _clients:
        _clients[base_url] = OpenAI(api_key=os.environ[key_env], base_url=base_url)
    return _clients[base_url]


def answer_question(g, client: OpenAI | None, question: str, hops: int = 2, retrieval: str = "harmonic") -> dict:
    """retrieval: "harmonic" (эмбеддинги целых документов из обученного
    чекпоинта) или "keyword" (baseline, наивное совпадение слов).
    hops=2 (пробовал урезать до 1 -- сломал главный механизм графа: нужные
    факты часто находятся не в документах-seed'ах напрямую, а через общий
    узел вроде "медь", связывающий seed-документ с фактом из документа,
    который сам по себе не попал в top-k по сходству. Обрезку текста при
    большом подграфе решает сортировка по релевантности ниже, не hops=1.
    Возвращает {seeds, subgraph_nodes, subgraph_edges, sources, answer}
    либо {seeds: [], answer: None, reason: "..."} если графу нечего сказать.
    Используется и CLI (main), и веб-приложением."""
    seeds, doc_scores, retrieval_used, retrieval_warning = retrieve_seed_nodes_auto(
        g, question, preferred=retrieval
    )
    if not seeds:
        return {
            "seeds": [],
            "answer": None,
            "reason": "Ничего не найдено в графе.",
            "doc_scores": doc_scores,
            "retrieval": retrieval_used,
            "retrieval_warning": retrieval_warning,
        }

    sub = neighborhood(g, seeds, hops=hops)
    facts_text, sources = facts_to_text(g, sub)
    if not facts_text:
        return {"seeds": [g.nodes[s]["canonical_name"] for s in seeds], "answer": None,
                "reason": "В окрестности найденных узлов нет связей (изолированные упоминания).",
                "retrieval": retrieval_used, "retrieval_warning": retrieval_warning}

    resp = None
    last_err = None
    used_model = None
    for provider_name, base_url, key_env, model in PROVIDERS:
        if key_env not in os.environ:
            continue
        try:
            c = _client_for(base_url, key_env)
            resolved_model = model.replace("__FOLDER__", os.environ.get("YANDEX_FOLDER_ID", ""))
            resp = c.chat.completions.create(
                model=resolved_model,
                # было 400000: yandexgpt/latest держит 32K токенов контекста,
                # на вопросах с большим подграфом такой объём фактов не влезал,
                # запрос отклонялся и цепочка падала на медленные фоллбэки
                # (ответ за минуту вместо секунд). 80000 символов русского
                # текста ~= 22-27K токенов -- влезает с запасом под ответ.
                messages=[{"role": "user", "content": ANSWER_PROMPT.format(question=question, facts=facts_text[:80000])}],
                temperature=0.2,
                max_tokens=4000,  # gpt-oss на Yandex -- reasoning-модель, съедает бюджет
                                  # на скрытые раздумья, при малом max_tokens content=None
            )
            if not resp.choices[0].message.content:
                last_err = RuntimeError(f"{provider_name}:{resolved_model} вернул пустой content")
                resp = None
                continue
            used_model = f"{provider_name}:{resolved_model}"
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            resp = None
            continue
    if resp is None:
        if last_err is None:
            llm_reason = "не задан ни один ключ: YANDEX_API_KEY, OPENROUTER_API_KEY или CEREBRAS_API_KEY"
        else:
            llm_reason = str(last_err)[:200]
        warning = f"LLM-синтез сейчас недоступен: {llm_reason}."
        if retrieval_warning:
            warning = f"{retrieval_warning}\n\n{warning}"
        return {
            "seeds": [g.nodes[s]["canonical_name"] for s in seeds],
            "subgraph_nodes": sub.number_of_nodes(),
            "subgraph_edges": sub.number_of_edges(),
            "sources": sorted(sources),
            "answer": f"{warning}\n\nНайденные факты графа:\n{facts_preview(facts_text)}",
            "retrieval": retrieval_used,
            "retrieval_warning": retrieval_warning,
            "doc_scores": doc_scores,
            "model": None,
            "llm_error": llm_reason,
            "graph": subgraph_to_viz(g, sub),
        }
    return {
        "seeds": [g.nodes[s]["canonical_name"] for s in seeds],
        "subgraph_nodes": sub.number_of_nodes(),
        "subgraph_edges": sub.number_of_edges(),
        "sources": sorted(sources),
        "answer": resp.choices[0].message.content,
        "retrieval": retrieval_used,
        "retrieval_warning": retrieval_warning,
        "doc_scores": doc_scores,
        "model": used_model,
        "graph": subgraph_to_viz(g, sub),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print('Использование: uv run python3 src/query/ask.py "твой вопрос" [harmonic|keyword]')
        sys.exit(1)
    question = sys.argv[1]
    retrieval = sys.argv[2] if len(sys.argv) > 2 else "harmonic"

    client = get_client()
    g = load_graph()
    result = answer_question(g, client, question, retrieval=retrieval)

    print(f"[{retrieval}] Найдено узлов-затравок: {len(result['seeds'])} -> {result['seeds'][:10]}")
    if result.get("doc_scores"):
        print("Топ документов по сходству:")
        for f, s in result["doc_scores"][:5]:
            print(f"  {s:.3f}  {f}")
    if result["answer"] is None:
        print(result.get("reason", "(без ответа, причина не указана)"))
        return
    print(f"Подграф: {result['subgraph_nodes']} узлов, {result['subgraph_edges']} рёбер, "
          f"{len(result['sources'])} источников\n")
    print("=== ОТВЕТ ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
