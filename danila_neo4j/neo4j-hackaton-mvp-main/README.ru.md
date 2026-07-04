# Neo4j Hackathon MVP

[English version](README.md)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20DB-4581C3)
![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

Это backend/data engineering MVP для хакатона. Сервис принимает уже структурированные факты о материалах в JSON, валидирует их через Pydantic, загружает в Neo4j knowledge graph и отдает graph-based рекомендации через FastAPI.

Важно: этот репозиторий пока не отвечает за LLM extraction. Предполагается, что отдельный upstream LLM/ML слой извлекает данные из статей или другого сырого текста и передает сюда готовый структурированный JSON.

## Архитектура

```text
JSON facts
↓
Pydantic validation
↓
IngestionService
↓
GraphLoader
↓
Neo4j Knowledge Graph
↓
GraphReader
↓
FastAPI API
```

Основные части проекта:

- `src/models.py` описывает Pydantic-схему для статей, экспериментов, методов, материалов, свойств, KPI и material facts.
- `src/parsers/json_parser.py` читает JSON и валидирует записи как `MaterialFact`.
- `src/ingestion_service.py` управляет процессом загрузки факта в граф.
- `src/graph_loader.py` создает узлы и связи в Neo4j.
- `src/graph_reader.py` читает evidence paths для запросов по KPI.
- `src/api.py` содержит FastAPI-приложение.
- `main.py` загружает demo data из `data/extracted/material_facts.json`.

## Модель графа

```text
Article -[:DESCRIBES]-> Experiment
Experiment -[:USES]-> Method
Experiment -[:TESTS]-> Material
Material -[:HAS_PROPERTY {value, unit}]-> Property
Property -[:AFFECTS]-> KPI
```

Основной evidence path:

```text
Article -> Experiment -> Material -> Property -> KPI
```

Идея такая: API может не просто вернуть материал, а показать, через какую статью, эксперимент и свойство этот материал связан с нужным KPI.

## Установка и запуск

Создайте и активируйте virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

Создайте `.env` из готового примера:

```bash
cp .env.example .env
```

Значения по умолчанию для локального запуска:

```env
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

Запустите Neo4j через Docker Compose:

```bash
docker compose up -d
```

Инициализируйте constraints в Neo4j:

```bash
docker compose exec -T neo4j cypher-shell -u neo4j -p password < scripts/init_constraints.cypher
```

Загрузите demo data:

```bash
python main.py
```

Запустите API:

```bash
uvicorn src.api:app --reload
```

API будет доступен по адресу `http://127.0.0.1:8000`.

## API Endpoints

### `GET /health`

Проверка, что API запущен.

Пример ответа:

```json
{
  "status": "ok"
}
```

### `GET /materials-for-kpi?name=Efficiency`

Возвращает материалы, связанные с нужным KPI, и evidence path для каждой найденной связи.

Пример ответа:

```json
[
  {
    "article": "Conductivity study of copper alloys",
    "experiment": "Copper conductivity test",
    "method": "Lab conductivity measurement",
    "material": "Copper Alloy",
    "property": "Conductivity",
    "kpi": "Efficiency",
    "value": 0.92,
    "unit": "score",
    "evidence_path": "Conductivity study of copper alloys -> Copper conductivity test -> Copper Alloy -> Conductivity -> Efficiency"
  },
  {
    "article": "Conductivity study of aluminum alloys",
    "experiment": "Aluminum conductivity test",
    "method": "Lab conductivity measurement",
    "material": "Aluminum Alloy",
    "property": "Conductivity",
    "kpi": "Efficiency",
    "value": 0.78,
    "unit": "score",
    "evidence_path": "Conductivity study of aluminum alloys -> Aluminum conductivity test -> Aluminum Alloy -> Conductivity -> Efficiency"
  }
]
```

### `GET /recommendations?name=Efficiency`

Возвращает лучший материал-кандидат по сохраненному значению `value`, а остальные найденные материалы кладет в `alternatives`.

Пример ответа:

```json
{
  "kpi": "Efficiency",
  "best_candidate": {
    "material": "Copper Alloy",
    "score": 0.92,
    "reason": "Copper Alloy is linked to Efficiency through Conductivity based on Copper conductivity test from Conductivity study of copper alloys.",
    "article": "Conductivity study of copper alloys",
    "experiment": "Copper conductivity test",
    "method": "Lab conductivity measurement",
    "evidence_path": "Conductivity study of copper alloys -> Copper conductivity test -> Copper Alloy -> Conductivity -> Efficiency"
  },
  "alternatives": [
    {
      "material": "Aluminum Alloy",
      "score": 0.78,
      "property": "Conductivity",
      "evidence_path": "Conductivity study of aluminum alloys -> Aluminum conductivity test -> Aluminum Alloy -> Conductivity -> Efficiency"
    }
  ]
}
```

## Почему Neo4j

Neo4j здесь используется потому, что главная ценность MVP находится в связях между сущностями. Рекомендация материала должна быть объяснимой: из какой статьи она пришла, в каком эксперименте проверялась, какое свойство связано с каким KPI.

В табличной модели такие цепочки быстро становятся набором join-запросов. В графовой модели путь `Article -> Experiment -> Material -> Property -> KPI` является естественной частью данных, его удобно искать, сортировать и отдавать наружу как evidence.

## Текущий статус

Реализовано:

- Загрузка JSON из `data/extracted/material_facts.json`.
- Pydantic-валидация структурированных material facts.
- Создание узлов и связей в Neo4j.
- Constraints на уникальные идентификаторы узлов.
- Поиск evidence paths для material recommendations по KPI.
- FastAPI endpoints для health check, поиска материалов по KPI и простой рекомендации.

Пока не реализовано:

- LLM/ML extraction из статей или сырого текста.
- Authentication/authorization.
- Автоматические тесты.
- Сложная модель ранжирования кроме сортировки по сохраненному `value`.
- Production deployment configuration.

## Роль в хакатоне

Этот репозиторий закрывает Data/Graph Engineering layer.

Upstream LLM/ML extraction layer должен отдавать структурированный JSON. Этот сервис отвечает за валидацию схемы, загрузку данных в граф, получение evidence paths и API-ответы, которые можно использовать в demo UI, agent layer или другом downstream-сервисе.
