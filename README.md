<!-- nddev-public-hackathon -->
# hack-2026-nornickel-knot

Nornickel AI Hack 2026, Scientific Knot track: Harmonic Labs research knowledge graph with source-backed answers.

[NDDev OpenNetwork](https://github.com/NDDev-Technocracy-OpenNetwork) · [Repository](https://github.com/rldyourmnd/hack-2026-nornickel-knot)

This repository is a hackathon prototype or supporting tool. Follow the project-specific setup below; cloud integrations require your own configuration and credentials.

---

<p align="center">
  <img src="https://i.ibb.co/hwjxFJq/harmonic-labs-nornickel-logo.jpg" alt="Harmonic Labs · Nornickel Scientific Knot" width="100%" />
</p>

## Nornickel Scientific Knot

Живая карта знаний R&D для горно-металлургической отрасли: система превращает архив отчётов, статей, презентаций и экспериментальных материалов в типизированный граф знаний, а затем отвечает на вопросы естественным языком с источниками и страницами первоисточников.

Проект подготовлен командой **Harmonic Labs** для **Норникель AI Hack 2026**, трек **«Научный клубок»**.

## Что внутри

- **82 706 узлов** графа знаний: материалы, процессы, оборудование, свойства, эксперименты, публикации, эксперты и объекты.
- **159 532 связи** с числовыми значениями, единицами измерения, условиями и ссылками на страницы источников.
- **Harmonic retrieval**: собственная SSM-модель Harmonic строит document-level embeddings и находит смысловые затравки для графового поиска.
- **YandexGPT synthesis**: финальный ответ собирается только по фактам графа, без свободной генерации из головы.
- **Neo4j-ready export**: контракт данных `documents / chunks / entities / mentions / relations` для промышленной графовой БД.
- **Web UI**: чат, интерактивный граф, загрузка PDF, вкладка pitch и презентационные материалы.

## Возможности

- Вопросы естественным языком с многопараметрическим контекстом: материал, процесс, условия, числа, география.
- Графовая навигация в стиле Neo4j/Obsidian: zoom, pan, drag узлов, hover-подсветка соседей, раскрытие связей вокруг запроса.
- Живое пополнение графа: пользователь загружает PDF, документ извлекается в ту же онтологию и добавляется без перезапуска.
- Проверяемость: у каждого факта есть источник, страницы, confidence и условия эксперимента.
- Fallback-режим: если checkpoint Harmonic отсутствует, backend не падает и откатывается на keyword retrieval.

## Быстрый запуск

```bash
./run_webapp.sh
```

Откройте сайт:

```text
http://127.0.0.1:8000
```

Скрипт запуска сначала использует локальный `.venv`, затем `uv`, затем системный `python3`.

## Переменные окружения

Для LLM-синтеза нужен хотя бы один провайдер. Пример лежит в `.env.example`:

```env
YANDEX_API_KEY=
YANDEX_FOLDER_ID=
OPENROUTER_API_KEY=
CEREBRAS_API_KEY=
```

Основной официальный стек для демо — **YandexGPT**.

## Runtime-артефакты

В репозитории есть подготовленный граф и document embeddings. Для полноценного Harmonic retrieval нужен checkpoint:

```text
data/harmonic_runs/harmonic_corpus_final.pt
```

Файл крупный и не хранится в git. Если checkpoint отсутствует, сайт продолжит работать с keyword retrieval и явно покажет fallback.

## Web endpoints

| Endpoint | Назначение |
| --- | --- |
| `/` | Основной сайт: чат, граф, pitch, Harmonic page |
| `/api/stats` | Статистика графа |
| `/api/ask` | Вопрос → retrieval → подграф → LLM-ответ |
| `/api/graph/browse` | Обзор графа по типам и строковому фильтру |
| `/api/graph/discover` | Harmonic-затравки и раскрытие соседей |
| `/api/graph/node` | Карточка узла и факты вокруг него |
| `/api/upload` | Загрузка PDF в живой граф |
| `/docs/pitch/deck.html` | HTML-презентация |
| `/docs/pitch/nornickel_harmonic_labs_pitch.pptx` | PPTX-презентация |

## Презентация

Готовые материалы находятся в `docs/pitch`:

- `deck.html` — цельная HTML-презентация в теме сайта;
- `slides/slide_01.html` ... `slides/slide_10.html` — отдельный HTML для каждого слайда;
- `nornickel_harmonic_labs_pitch.pptx` — PowerPoint deck;
- `build_pitch_deck.py` — генератор HTML/PPTX из единого набора данных.

Пересборка презентации:

```bash
.venv/bin/python docs/pitch/build_pitch_deck.py
```

## Архитектура

```text
Raw files
  → parsing + OCR
  → chunking
  → Qwen2.5 fact extraction
  → normalized entities and relations
  → NetworkX knowledge graph
  → Harmonic / keyword retrieval
  → neighborhood expansion
  → YandexGPT answer with sources
  → Web graph visualization
```

### Основные модули

```text
src/graph/              построение, хранение и экспорт графа
src/query/ask.py        retrieval + LLM answer synthesis
src/harmonic_train/     Harmonic model, embeddings, training scripts
src/webapp/             FastAPI app and ingestion pipeline
src/webapp/static/      frontend: chat, graph explorer, pitch UI
docs/pitch/             HTML/PPTX presentation assets
danila_neo4j/           Neo4j MVP/export materials
```

## Data contract

Графовые данные экспортируются в формат, пригодный для Neo4j и downstream-систем:

- `documents.jsonl`
- `chunks.jsonl`
- `entities.jsonl`
- `entity_mentions.jsonl`
- `relations.jsonl`
- `harmonic_documents.jsonl`
- `extraction_report.csv`

## Проверки

Минимальные smoke checks после запуска:

```bash
curl http://127.0.0.1:8000/api/stats
curl http://127.0.0.1:8000/docs/pitch/deck.html
```

Ожидаемые масштабы графа:

```text
nodes: 82706
edges: 159532
```

## Команда

**Harmonic Labs**

- собственная SSM-модель Harmonic;
- граф знаний R&D;
- официальный LLM-стек РФ через YandexGPT;
- открытый Python/FastAPI/PyTorch/networkx стек.

---

<p align="center"><strong>Harmonic Labs — мостик между прошлым опытом и будущими открытиями.</strong></p>
