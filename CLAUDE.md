# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

API key in `.env` (loaded automatically via `python-dotenv`):
```
OPENROUTER_API_KEY=sk-or-...
```

## Running Tests

```bash
source venv/bin/activate

# Социологические гипотезы (основной тест)
python3 test_sociology.py

# Прямой тест OpenRouter + unit-тест get_llm()
python3 test_openrouter.py
```

## Architecture

POPPER — фреймворк для автоматической валидации гипотез методом последовательной фальсификации (по Попперу). LLM-агент разрабатывает статистические тесты, выполняет Python-код на данных и возвращает p-value.

### Основной поток

```
Popper.validate(hypothesis)
  → SequentialFalsificationTest.go()        # agent.py
      → test_proposal_agent                 # придумывает тест (LLM)
      → falsification_test_react_agent      # пишет и выполняет код (ReactAgent)
      → relevance_checker                   # фильтр нерелевантных тестов (LLM)
      → sequential_testing                  # E-value / p-value агрегация
      → summarizer                          # итоговый вердикт (LLM)
```

### Ключевые файлы

| Файл | Роль |
|---|---|
| `popper/popper.py` | Публичный API: `Popper` класс (register_data → configure → validate) |
| `popper/agent.py` | Вся логика агентов и LangGraph-граф |
| `popper/utils.py` | `get_llm()` — фабрика LLM; загрузчики данных |
| `popper/react_agent.py` | `ReactAgent` — выполняет Python-код на данных |
| `popper/llm/custom_model.py` | `CustomChatModel` — ChatOpenAI-совместимая обёртка для локальных/OpenRouter моделей |

### Загрузчики данных (`loader_type`)

- `'bio'` / `'bio_selected'` — биологическая база (~2.27 ГБ, скачивается с Harvard Dataverse)
- `'custom'` — любые CSV/PKL файлы из указанной папки (для социологии, экономики и т.д.)
- `'discovery_bench'` — мультидоменный бенчмарк из статьи (требует `metadata`)

### Поддержка моделей в `get_llm()` и `ReactAgent`

Определяется по префиксу имени модели:
- `claude-*` → Anthropic API
- `gpt-*` / `o1*` → OpenAI API
- `*/` (содержит `/`) → **OpenRouter** (добавлено в этом репо; `base_url=https://openrouter.ai/api/v1`; текущая модель: `deepseek/deepseek-v3.2`, reasoning включён для всех OpenRouter-моделей)
- иначе → локальная модель (требует `port`)

### Внесённые изменения (не из оригинального репо)

1. **`popper/utils.py`** — добавлена ветка `OpenRouter` в `get_llm()` с `OPENROUTER_BASE_URL`
2. **`popper/react_agent.py`** — добавлена ветка `openrouter` в `ReactAgent.__init__` и `get_model()`
3. **`popper/llm/custom_model.py`** — исправлен парсинг `tool_calls.args` (JSON-строка → dict)
4. **`popper/popper.py`** — исправлены два бага:
   - `bio_database` скачивается только для `loader_type='bio'/'bio_selected'`
   - добавлен параметр `domain` в публичный метод `configure()` (пробрасывается в `SequentialFalsificationTest.configure()`)
5. **`web_app.py`** — новый файл: веб-интерфейс Gradio с загрузкой CSV, превью данных, вводом гипотезы, выбором домена (sociology, biology, economics и др.), настройками alpha/max_tests и стримингом лога агента

### Запуск веб-интерфейса

```bash
source venv/bin/activate
python3 web_app.py
# Открыть http://localhost:7860
```
