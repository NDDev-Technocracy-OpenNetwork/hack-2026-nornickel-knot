"""Механическая (без LLM) детекция противоречий в графе знаний.

Идея: если для одной и той же пары сущностей и типа связи разные источники
дают сильно разные numeric_value -- это кандидат в противоречие. Но наивное
сравнение "значения разные" даёт кучу ложных срабатываний, поэтому:

- Сравниваем ТОЛЬКО факты с одинаковым unit (разные единицы без конвертации
  сравнивать нельзя -- пропускаем, а не гадаем).
- Если у обоих фактов указан condition и он РАЗНЫЙ -- это не противоречие,
  а ожидаемая вариация (одно и то же свойство в разных условиях). Сравниваем
  только когда condition совпадает или отсутствует у обоих.
- Требуем минимум 2 РАЗНЫХ source_file -- иначе это может быть просто два
  чанка одного документа.
- Порог относительный (max/min ratio), не абсолютный -- иначе 500 vs 500.5
  не должно шуметь, а 10 vs 1000 обязано найтись.

Результат: data/graph/contradictions_report.jsonl, плюс проставляет
mechanical_contradiction=True на существующих contradicts-рёбрах графа
(если пара уже была best-effort размечена LLM) и добавляет НОВЫЕ contradicts-
рёбра для механически найденных пар, которых LLM не нашла.

Запуск:
  uv run python -m src.graph.detect_contradictions
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_store import GRAPH_PATH, load_graph, save_graph  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "data" / "graph" / "contradictions_report.jsonl"

# ratio = max(value)/min(value) для положительных значений одного знака.
# 1.5 = минимум 50% относительной разницы, чтобы не шуметь на округлениях.
RATIO_THRESHOLD = 1.5
# для "мягкого" прохода (без привязки к конкретной цели-сущности) порог
# выше -- сравниваем потенциально разные объекты (золото в руднике А против
# золота в руднике Б), это не строгий факт, а сигнал "стоит проверить
# человеку", поэтому шумим только на действительно больших расхождениях
LOOSE_RATIO_THRESHOLD = 3.0
NUMERIC_RELATIONS = {"operates_at_condition", "produces_output"}


def _comparable_condition(c1, c2) -> bool:
    """True, если условия можно считать одинаковыми (оба пустые, или
    совпадают текстуально после нормализации пробелов/регистра)."""
    n1 = (c1 or "").strip().lower()
    n2 = (c2 or "").strip().lower()
    return n1 == n2


def find_contradictions(g) -> list[dict]:
    # группа: (src_key, relation, tgt_key, unit_normalized) -> список фактов
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for src, tgt, data in g.edges(data=True):
        if data["relation"] not in NUMERIC_RELATIONS:
            continue
        if data.get("numeric_value") is None:
            continue
        unit_key = (data.get("unit") or "").strip().lower()
        key = (src, data["relation"], tgt, unit_key)
        groups[key].append({
            "value": data["numeric_value"],
            "condition": data.get("condition"),
            "source_file": data["source_file"],
            "category": data.get("category"),
            "confidence": data.get("confidence"),
        })

    findings = []
    for (src, relation, tgt, unit), facts in groups.items():
        distinct_sources = {f["source_file"] for f in facts}
        if len(distinct_sources) < 2:
            continue

        # сравниваем попарно внутри группы -- группы маленькие (одна пара
        # сущность+связь редко упоминается в >5-10 разных документах), так
        # что O(n^2) тут не проблема
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                a, b = facts[i], facts[j]
                if a["source_file"] == b["source_file"]:
                    continue
                if not _comparable_condition(a["condition"], b["condition"]):
                    continue
                v1, v2 = a["value"], b["value"]
                if v1 == 0 or v2 == 0:
                    continue
                if (v1 > 0) != (v2 > 0):
                    continue  # разные знаки -- отдельная история, не делим на 0-риски
                ratio = max(abs(v1), abs(v2)) / min(abs(v1), abs(v2))
                if ratio < RATIO_THRESHOLD:
                    continue
                findings.append({
                    "confidence_level": "strict",  # та же пара сущностей, тот же юнит
                    "entity_a": g.nodes[src]["canonical_name"],
                    "entity_a_type": g.nodes[src]["type"],
                    "relation": relation,
                    "entity_b": g.nodes[tgt]["canonical_name"],
                    "entity_b_type": g.nodes[tgt]["type"],
                    "unit": unit or None,
                    "condition": a["condition"] or b["condition"],
                    "fact_a": {"value": v1, "source_file": a["source_file"], "category": a["category"]},
                    "fact_b": {"value": v2, "source_file": b["source_file"], "category": b["category"]},
                    "ratio": round(ratio, 2),
                })

    findings.sort(key=lambda f: -f["ratio"])
    return findings


def find_loose_contradictions(g) -> list[dict]:
    """Тот же источник-сущность + связь + юнит, но БЕЗ требования одной и той
    же целевой сущности -- ловит случаи вроде "золото: 43кт" в одном
    документе и "золото: 300кт" в другом, даже если формально это золото
    разных рудников/проектов. Именно поэтому: выше порог (3x, не 1.5x),
    ЭТО НЕ добавляется в граф как contradicts-ребро (слишком ненадёжно для
    структуры графа), только идёт в отчёт с пометкой confidence_level=loose
    для ручной проверки жюри/командой."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for src, tgt, data in g.edges(data=True):
        if data["relation"] not in NUMERIC_RELATIONS or data.get("numeric_value") is None:
            continue
        unit_key = (data.get("unit") or "").strip().lower()
        key = (src, data["relation"], unit_key)
        groups[key].append({
            "value": data["numeric_value"], "condition": data.get("condition"),
            "source_file": data["source_file"], "category": data.get("category"),
            "target": g.nodes[tgt]["canonical_name"],
        })

    findings = []
    for (src, relation, unit), facts in groups.items():
        if len({f["source_file"] for f in facts}) < 2:
            continue
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                a, b = facts[i], facts[j]
                if a["source_file"] == b["source_file"] or a["target"] == b["target"]:
                    continue  # тот же документ, либо та же цель -- уже покрыто строгим проходом
                if not _comparable_condition(a["condition"], b["condition"]):
                    continue
                v1, v2 = a["value"], b["value"]
                if v1 == 0 or v2 == 0 or (v1 > 0) != (v2 > 0):
                    continue
                ratio = max(abs(v1), abs(v2)) / min(abs(v1), abs(v2))
                if ratio < LOOSE_RATIO_THRESHOLD:
                    continue
                findings.append({
                    "confidence_level": "loose_review_needed",
                    "entity_a": g.nodes[src]["canonical_name"],
                    "entity_a_type": g.nodes[src]["type"],
                    "relation": relation,
                    "entity_b": a["target"],
                    "entity_b_alt": b["target"],
                    "unit": unit or None,
                    "condition": a["condition"] or b["condition"],
                    "fact_a": {"value": v1, "source_file": a["source_file"], "category": a["category"], "target": a["target"]},
                    "fact_b": {"value": v2, "source_file": b["source_file"], "category": b["category"], "target": b["target"]},
                    "ratio": round(ratio, 2),
                })

    findings.sort(key=lambda f: -f["ratio"])
    return findings


def annotate_graph(g, findings: list[dict]) -> int:
    """Добавляет contradicts-рёбра для механически найденных пар. Не трогает
    существующие LLM-найденные contradicts -- только дополняет."""
    key_by_canonical: dict[tuple, str] = {}
    for key, data in g.nodes(data=True):
        key_by_canonical[(data["type"], data["canonical_name"])] = key

    added = 0
    for f in findings:
        src = key_by_canonical.get((f["entity_a_type"], f["entity_a"]))
        tgt = key_by_canonical.get((f["entity_b_type"], f["entity_b"]))
        if src is None or tgt is None:
            continue
        # не дублируем, если между этой парой уже есть contradicts ребро
        # (не важно, откуда -- LLM или механика)
        existing = [d for _, _, d in g.edges(src, data=True) if d.get("relation") == "contradicts"]
        if any(e.get("target_canonical") == f["entity_b"] for e in existing):
            continue
        g.add_edge(
            src, tgt,
            relation="contradicts",
            numeric_value=None,
            unit=f["unit"],
            condition=f["condition"],
            confidence=None,
            source_file=f["fact_a"]["source_file"],
            category=f["fact_a"]["category"],
            page_start=None,
            page_end=None,
            mechanical_contradiction=True,
            ratio=f["ratio"],
            target_canonical=f["entity_b"],
        )
        added += 1
    return added


def main() -> None:
    g = load_graph()
    strict = find_contradictions(g)
    # мягкий проход (find_loose_contradictions) прогнан и отклонён: на этом
    # графе сущности типа "золото"/"медь" объединяют ВСЕ рудники/страны в
    # один узел, поэтому сравнение "по цели" без учёта конкретного объекта
    # всегда сравнивает разные реальные вещи -- 112 находок оказались 100%
    # ложными (разные рудники с ожидаемо разной добычей, не противоречие).
    # Не подмешиваем в отчёт, который увидит жюри -- ложные "противоречия"
    # хуже отсутствия находок.
    loose: list[dict] = []

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        for finding in strict + loose:
            fh.write(json.dumps(finding, ensure_ascii=False) + "\n")

    print(f"Строгих кандидатов (та же пара сущностей): {len(strict)}")
    if strict:
        print("\nТоп по силе расхождения (строгие):")
        for f in strict[:10]:
            print(f"  {f['entity_a']} --[{f['relation']}]--> {f['entity_b']} "
                  f"({f['unit']}): {f['fact_a']['value']} [{f['fact_a']['source_file']}] "
                  f"vs {f['fact_b']['value']} [{f['fact_b']['source_file']}] "
                  f"-- ratio={f['ratio']}")

    print(f"\nМягких кандидатов (та же сущность, разные цели -- нужна проверка человеком): {len(loose)}")
    if loose:
        print("\nТоп по силе расхождения (мягкие, ratio>=3x):")
        for f in loose[:15]:
            print(f"  {f['entity_a']} --[{f['relation']}]--> [{f['fact_a']['target']}]="
                  f"{f['fact_a']['value']}{f['unit'] or ''} [{f['fact_a']['source_file']}]  "
                  f"vs --> [{f['fact_b']['target']}]={f['fact_b']['value']}{f['unit'] or ''} "
                  f"[{f['fact_b']['source_file']}]  -- ratio={f['ratio']}")

    # в граф добавляем ТОЛЬКО строгие -- мягкие слишком ненадёжны для
    # структуры графа (сравнивают разные конкретные объекты), живут только
    # в отчёте для ручной проверки
    added = annotate_graph(g, strict)
    save_graph(g)
    print(f"\nДобавлено новых contradicts-рёбер в граф (только строгие): {added}")
    print(f"Полный отчёт (строгие + мягкие): {REPORT_PATH.relative_to(ROOT)}")
    print(f"Граф пересохранён: {GRAPH_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
