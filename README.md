# POPPER — Проверка гипотез (форк с веб-интерфейсом)

Форк оригинального [snap-stanford/POPPER](https://github.com/snap-stanford/POPPER) с добавлением:

- **Поддержки OpenRouter** — единый API-ключ для доступа к сотням моделей (Gemini, Claude, GPT-4o, Llama и др.)
- **Веб-интерфейса** (`web_app.py`) — загрузка CSV, ввод гипотезы, стриминг лога агента и финальный вердикт в браузере
- **Docker-деплоя** — готовый `Dockerfile` и `docker-compose.yml`
- **Нескольких багфиксов** — парсинг `tool_calls`, параметр `domain`, загрузка bio-датасета

---

## Быстрый старт

### Локально

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=sk-or-..." > .env
python3 web_app.py
# Открыть http://localhost:7860
```

### Docker

```bash
docker compose up --build
# Открыть http://localhost:7860
```

Переменная окружения `OPENROUTER_API_KEY` должна быть задана в `.env` (локально) или в настройках хостинга.

---

## Как использовать веб-интерфейс

1. Загрузите CSV-файл с данными (минимум 30 строк)
2. Введите гипотезу на любом языке
3. Выберите домен (социология, экономика и т.д.)
4. Нажмите **«Проверить»** — агент разработает статистические тесты и вынесет вердикт

---

## Python API

```python
from popper import Popper

agent = Popper(llm="google/gemini-2.5-pro-preview", api_key="sk-or-...")
agent.register_data(data_path="path/to/csvs", loader_type="custom")
agent.configure(alpha=0.1, max_num_of_tests=3, domain="sociology")
results = agent.validate("Ваша гипотеза")
```

Модель задаётся по имени: `provider/model` → OpenRouter, `claude-*` → Anthropic, `gpt-*` → OpenAI.

---

Оригинальная статья: [arxiv.org/abs/2502.09858](https://arxiv.org/abs/2502.09858)
