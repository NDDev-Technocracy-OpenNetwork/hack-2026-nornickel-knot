"""Batch-извлечение фактов через self-hosted Qwen на H100 (vLLM offline batch --
не сервер, один процесс грузит модель и прогоняет все чанки разом).

Решает проблему дневных лимитов бесплатных API (Cerebras 1M ток/день,
OpenRouter free-tier): свой GPU, без лимита на запросы, вся пачка чанков
за один прогон vLLM с continuous batching.

Запуск:
  modal run src/extract/modal_vllm_extract.py
"""

import json
import pathlib

import modal

app = modal.App("nornikel-vllm-extract")
# Постоянное хранилище на стороне Modal -- переживает обрыв ЛОКАЛЬНОГО
# клиента полностью (в отличие от .spawn()+call.get(), где результат всё
# равно нужно ДОЖДАТЬСЯ одним вызовом). Пишем сюда после каждой под-пачки,
# читаем оттуда отдельно, независимо от того, жив ли исходный процесс.
results_dict = modal.Dict.from_name("nornikel-extract-results", create_if_missing=True)
# cwd, не __file__.parents -- этот модуль исполняется и локально (cwd=корень
# проекта, т.к. modal run запускается оттуда), и удалённо в контейнере Modal,
# где файл лежит на другой глубине (/root/modal_vllm_extract.py) и parents[2]
# уже не существует. Remote-сторона (run_batch) эти пути не использует вовсе.
ROOT = pathlib.Path.cwd()

SOURCE_ROOT = ROOT / "data" / "parsed"
DEST_ROOT = ROOT / "data" / "extracted"
# было 12000: контрольная проверка на реальных вопросах показала, что это
# обрезает документы задолго до релевантного контента (медиана документа
# в корпусе -- 79000 символов, у крупных до 800000+). 120000 покрывает
# медианный документ целиком; самые длинные всё ещё частично обрезаются --
# честно задокументированное, не скрытое ограничение.
MAX_CHARS_PER_DOC = 120000
CHUNK_SIZE = 3000

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.6.6", "transformers==4.46.3", "tokenizers==0.20.3", "pydantic")
)

# Скопировано из extract.py / schema.py напрямую -- не импортирую,
# чтобы local_entrypoint не тянул openai-клиент, который тут не нужен.

ONTOLOGY_PROMPT = """Ты извлекаешь структурированные факты из технического текста
по горно-металлургической отрасли (добыча, обогащение, гидро- и пирометаллургия)
для карты знаний R&D. Текст может быть на русском или английском.

Извлекай ТОЛЬКО то, что явно есть в тексте. Не выдумывай факты и цифры.
Не создавай отдельную сущность Publication для каждого номера ссылки из списка
литературы, если о ней в тексте больше ничего не сказано -- это мусор, не факт.

Типы сущностей: Material, Process, Equipment, Property, Experiment,
Publication, Expert, Facility.

Типы связей: uses_material, operates_at_condition, produces_output,
described_in, validated_by, contradicts, affiliated_with.

affiliated_with -- эксперт или исследовательская команда работает в/принадлежит
организации/лаборатории/институту (например: "Вострикова Н.М." --[affiliated_with]--> "Институт Гипроникель").
Используй это явно, когда текст называет организацию авторов/экспертов --
не оставляй такую связь висеть только на described_in.

Для canonical_name: нормализуй синонимы и переводы к одному виду
(например «электроэкстракция» и «electrowinning» -> «electrowinning»).

Если в куске текста нет явных фактов по этой онтологии -- верни пустые списки.

Верни строго JSON без пояснений, по схеме:
{{"entities": [{{"id": str, "type": str, "name": str, "canonical_name": str}}],
  "relations": [{{"source_id": str, "relation": str, "target_id": str,
                  "numeric_value": float|null, "unit": str|null,
                  "condition": str|null, "confidence": float}}]}}

Текст:
---
{chunk_text}
---
"""


def make_chunks(pages: list[dict], max_total_chars: int, chunk_size: int) -> list[dict]:
    combined = []
    total = 0
    for p in pages:
        if total >= max_total_chars:
            break
        remaining = max_total_chars - total
        text = p["text"][:remaining]
        if text.strip():
            combined.append((p["page"], text))
        total += len(text)

    chunks = []
    buf, buf_pages = "", []
    for page_num, text in combined:
        buf += text
        buf_pages.append(page_num)
        if len(buf) >= chunk_size:
            chunks.append({"text": buf, "page_start": buf_pages[0], "page_end": buf_pages[-1]})
            buf, buf_pages = "", []
    if buf.strip():
        chunks.append({"text": buf, "page_start": buf_pages[0], "page_end": buf_pages[-1]})
    return chunks


MAX_DOCS_PER_RUN = 1500  # 7B даёт ~2.76 валидных чанков/сек -- весь
# оставшийся корпус (~1008 документов, ~33000 чанков) укладывается в
# ~3.5 часа одним прогоном, лимит уже не нужен, оставлен с запасом сверху


def collect_pending_chunks():
    """Те же документы/чанки что и extract.py, но только уже НЕ извлечённые."""
    pending = []
    n_docs = 0
    for parsed_path in sorted(SOURCE_ROOT.rglob("*.json")):
        if n_docs >= MAX_DOCS_PER_RUN:
            break
        category_dir = DEST_ROOT / parsed_path.parent.name
        dest_path = category_dir / parsed_path.name
        if dest_path.exists():
            continue
        doc = json.loads(parsed_path.read_text(encoding="utf-8"))
        chunks = make_chunks(doc["pages"], MAX_CHARS_PER_DOC, CHUNK_SIZE)
        for i, chunk in enumerate(chunks):
            pending.append({
                "filename": doc["filename"], "category": doc["category"],
                "chunk_index": i, "page_start": chunk["page_start"], "page_end": chunk["page_end"],
                "text": chunk["text"],
            })
        n_docs += 1
    return pending


@app.function(gpu="H100", image=image, timeout=6 * 3600, max_containers=10)
def run_batch(items_json: str, worker_id: int = 0) -> str:
    from pydantic import BaseModel
    from typing import Literal, Optional
    from vllm import LLM, SamplingParams

    class Entity(BaseModel):
        id: str
        type: Literal["Material", "Process", "Equipment", "Property", "Experiment", "Publication", "Expert", "Facility"]
        name: str
        canonical_name: str

    class Relation(BaseModel):
        source_id: str
        relation: Literal["uses_material", "operates_at_condition", "produces_output", "described_in", "validated_by", "contradicts", "affiliated_with"]
        target_id: str
        numeric_value: Optional[float] = None
        unit: Optional[str] = None
        condition: Optional[str] = None
        confidence: Optional[float] = None

    class ExtractionResult(BaseModel):
        entities: list[Entity]
        relations: list[Relation]

    items = json.loads(items_json)
    print(f"Загружаю {MODEL_ID} ...", flush=True)
    # Промпт ~2060 токенов + max_tokens=2560 на ответ ~= 4620 -- подняли
    # max_model_len под это. Но с max_model_len=6144 vLLM сам печатает
    # реальный потолок: "Maximum concurrency for 6144 tokens per request:
    # 174.04x" -- мы стояли на 224, ВЫШЕ потолка (был риск повторить старый
    # затор с преэмпшенами). 160 -- с небольшим запасом ниже реального 174.
    llm = LLM(model=MODEL_ID, dtype="bfloat16", max_model_len=6144,
              gpu_memory_utilization=0.92, max_num_seqs=160)

    # было 4096: на 24000+ чанках разом это забивало KV-кэш (PreemptionMode.
    # RECOMPUTE, GPU толокся сам с собой) -- JSON-ответ (сущности+связи с
    # одного чанка ~3000 символов) реально помещается в разы меньше
    #
    # guided_decoding (JSON-схема через логит-маскирование) пробовали --
    # xgrammar несовместим по версии, lm-format-enforcer совместим, но
    # даёт огромный оверхед на токен (упало с ~300 ток/с до ~2 ток/с) --
    # цена за отказ от малого % провалов парсинга оказалась неприемлемой.
    # Остаёмся на обычном JSON + clean_json()-фоллбэке ниже.
    # 1536 резало JSON на полуслове у 7B (EOF while parsing a list) -- она
    # многословнее 32B в форматировании. Памяти теперь много (7B лёгкая),
    # можно позволить больше без риска повторить прежний KV-кэш затор.
    sampling = SamplingParams(temperature=0.1, max_tokens=2560)

    def clean_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        return text.strip()

    # Пользователь осознанно выбрал скорость важнее защиты от краха --
    # каждый новый generate() разгоняется по 1-2 минуты, много под-пачек
    # суммарно съедают заметное время на повторные разгоны. SUB_BATCH
    # выше общего числа чанков = фактически один вызов на всё сразу.
    # Механизм results_dict/run_id остаётся -- если где-то всё же наберётся
    # больше одной под-пачки (не в этот раз), защита сработает как раньше.
    SUB_BATCH = 100000
    run_id = items[0].get("run_id", "run") if items else "run"
    total_ok, total_failed, total_docs = 0, 0, set()
    for start_idx in range(0, len(items), SUB_BATCH):
        sub_items = items[start_idx:start_idx + SUB_BATCH]
        prompts = [ONTOLOGY_PROMPT.format(chunk_text=it["text"]) for it in sub_items]
        print(f"Под-пачка {start_idx}-{start_idx+len(sub_items)}/{len(items)}...", flush=True)
        outputs = llm.generate(prompts, sampling)

        by_doc: dict[str, dict] = {}
        ok, failed = 0, 0
        for it, out in zip(sub_items, outputs):
            raw = clean_json(out.outputs[0].text)
            try:
                parsed = ExtractionResult.model_validate_json(raw)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  parse FAILED: {it['filename']} chunk {it['chunk_index']}: {str(e)[:150]}", flush=True)
                continue
            ok += 1
            key = f"{it['category']}/{it['filename']}"
            entry = by_doc.setdefault(key, {"filename": it["filename"], "category": it["category"],
                                             "entities": [], "relations": []})
            provenance = {"source_file": it["filename"], "category": it["category"],
                          "chunk_index": it["chunk_index"], "page_start": it["page_start"], "page_end": it["page_end"]}
            for e in parsed.entities:
                entry["entities"].append({**e.model_dump(), "provenance": provenance})
            for rel in parsed.relations:
                entry["relations"].append({**rel.model_dump(), "provenance": provenance})

        sub_batch_key = f"{run_id}:{worker_id}:{start_idx}"
        results_dict[sub_batch_key] = json.dumps({"by_doc": by_doc, "ok": ok, "failed": failed}, ensure_ascii=False)
        total_ok += ok
        total_failed += failed
        total_docs |= set(by_doc.keys())
        print(f"  под-пачка готова: {ok} ок, {failed} провалов -- записано в results_dict[{sub_batch_key}]", flush=True)

    print(f"Воркер {worker_id} готов: {total_ok} чанков разобрано, {total_failed} не распарсились, "
          f"{len(total_docs)} документов.", flush=True)
    return json.dumps({"run_id": run_id, "worker_id": worker_id, "ok": total_ok,
                        "failed": total_failed, "n_docs": len(total_docs)})


CALL_ID_PATH = ROOT / "data" / "last_extract_call_id.txt"
RUN_ID_PATH = ROOT / "data" / "last_extract_run_id.txt"


def write_entries(by_doc: dict) -> int:
    """Пишет/дополняет data/extracted/*.json -- если документ уже частично
    записан (из предыдущей под-пачки этого же прогона), дописывает, а не
    затирает."""
    n = 0
    for key, entry in by_doc.items():
        stem = pathlib.Path(entry["filename"]).stem
        dest_path = DEST_ROOT / entry["category"] / f"{stem}.json"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            existing = json.loads(dest_path.read_text(encoding="utf-8"))
            existing["entities"].extend(entry["entities"])
            existing["relations"].extend(entry["relations"])
            entry = existing
        dest_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        n += 1
    return n


def harvest(run_id: str) -> None:
    """Читает все под-пачки прогона run_id из results_dict и пишет файлы.
    Не зависит от того, жив ли исходный .spawn()-вызов -- Dict персистентен
    сам по себе, можно запускать этот харвест сколько угодно раз позже."""
    prefix = f"{run_id}:"
    keys = [k for k in results_dict.keys() if k.startswith(prefix)]
    print(f"Найдено под-пачек в results_dict для {run_id}: {len(keys)}", flush=True)
    total_ok, total_failed, total_written = 0, 0, 0
    for k in sorted(keys):  # порядок неважен -- write_entries дописывает, не затирает
        chunk = json.loads(results_dict[k])
        total_ok += chunk["ok"]
        total_failed += chunk["failed"]
        total_written += write_entries(chunk["by_doc"])
    print(f"Готово: {total_ok} чанков разобрано, {total_failed} не распарсились, "
          f"{total_written} документов записано (из {len(keys)} под-пачек).", flush=True)


@app.local_entrypoint()
def main(call_id: str = "", harvest_run_id: str = ""):
    # harvest_run_id -- если .spawn()-вызов уже недоступен (истёк, потерян
    # call_id), но под-пачки успели записаться в results_dict, можно забрать
    # их отдельно в любой момент: modal run ...::main --harvest-run-id=XXX
    if harvest_run_id:
        harvest(harvest_run_id)
        return

    run_id = f"run_{int(__import__('time').time())}"
    # .remote() отменяется при обрыве локального клиента, даже под
    # `modal run --detach` -- сам Modal явно предупреждает об этом
    # ("Use .spawn() for detached or background work"). .spawn() отдаёт
    # call_id сразу и не зависит от того, жив ли клиент -- но теперь это
    # подстраховка №2: данные и так безопасны в results_dict по мере
    # готовности под-пачек, даже если .get() тут ничего не дождётся.
    N_WORKERS = 10  # 10 параллельных H100 вместо одной -- та же цена в
    # GPU-часах (Modal считает по факту использования), но в ~10x меньше
    # по настенным часам. Тот же приём что и для OCR-шага (build_corpus_v2.py).

    if call_id:
        from modal.functions import FunctionCall
        print(f"Переподключаюсь к вызову {call_id}...", flush=True)
        call = FunctionCall.from_id(call_id)
        result_json = call.get()
        run_id = json.loads(result_json)["run_id"]
        harvest(run_id)
        return

    pending = collect_pending_chunks()
    if not pending:
        print("Нечего извлекать -- всё уже готово.", flush=True)
        return
    for it in pending:
        it["run_id"] = run_id

    groups = [pending[i::N_WORKERS] for i in range(N_WORKERS)]
    groups = [(json.dumps(g, ensure_ascii=False), idx) for idx, g in enumerate(groups) if g]
    RUN_ID_PATH.write_text(run_id, encoding="utf-8")
    print(f"К извлечению: {len(pending)} чанков, {len(groups)} параллельных воркеров "
          f"(run_id={run_id}). Если что-то оборвётся -- забрать результат можно в любой "
          f"момент: modal run src/extract/modal_vllm_extract.py::main --harvest-run-id={run_id}",
          flush=True)

    for result_json in run_batch.starmap(groups, return_exceptions=True):
        if isinstance(result_json, Exception):
            print(f"  воркер упал: {result_json}", flush=True)
            continue
        r = json.loads(result_json)
        print(f"  воркер {r['worker_id']} завершён: {r['ok']} ок, {r['failed']} провалов, "
              f"{r['n_docs']} документов", flush=True)

    harvest(run_id)
