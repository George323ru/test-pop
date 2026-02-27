"""
Тестовый скрипт POPPER через OpenRouter + Google Gemini

Использование:
    export OPENROUTER_API_KEY="your-key-here"
    python test_openrouter.py

Что делает POPPER:
    - Принимает научную гипотезу в свободной форме
    - LLM-агент разрабатывает эксперименты для фальсификации гипотезы
    - Применяет статистическое тестирование с контролем ошибки I рода
    - Возвращает вердикт: подтверждена / фальсифицирована / недостаточно данных
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# -----------------------------------------------------------------------
# Раздел 1: Прямой тест OpenRouter + Gemini (без POPPER)
# -----------------------------------------------------------------------
def test_openrouter_direct():
    """Прямой вызов OpenRouter с reasoning — как в коде пользователя."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[ПРОПУЩЕН] OPENROUTER_API_KEY не задан")
        return

    print("=" * 60)
    print("Тест 1: Прямой вызов OpenRouter + Gemini с reasoning")
    print("=" * 60)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Первый вызов с reasoning
    response = client.chat.completions.create(
        model="deepseek/deepseek-v3.2",
        messages=[
            {
                "role": "user",
                "content": "How many r's are in the word 'strawberry'?"
            }
        ],
        extra_body={"reasoning": {"enabled": True}}
    )

    assistant_msg = response.choices[0].message
    print(f"Ответ: {assistant_msg.content}")
    if hasattr(assistant_msg, 'reasoning_details') and assistant_msg.reasoning_details:
        print(f"Reasoning: {assistant_msg.reasoning_details}")

    # Второй вызов — продолжение рассуждения
    messages = [
        {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
        {
            "role": "assistant",
            "content": assistant_msg.content,
            "reasoning_details": assistant_msg.reasoning_details  # передаём обратно
        },
        {"role": "user", "content": "Are you sure? Think carefully."}
    ]

    response2 = client.chat.completions.create(
        model="deepseek/deepseek-v3.2",
        messages=messages,
        extra_body={"reasoning": {"enabled": True}}
    )

    print(f"\nПодтверждение: {response2.choices[0].message.content}")
    print()


# -----------------------------------------------------------------------
# Раздел 2: Тест через POPPER с кастомными данными
# -----------------------------------------------------------------------
def test_popper_with_custom_data():
    """
    Тест POPPER с кастомными CSV-данными и гипотезой.
    Использует loader_type='custom' — не требует загрузки bio_database.
    """
    import pandas as pd
    import tempfile

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[ПРОПУЩЕН] OPENROUTER_API_KEY не задан")
        return

    print("=" * 60)
    print("Тест 2: POPPER с кастомными данными + OpenRouter Gemini")
    print("=" * 60)

    # Создаём простой синтетический датасет
    with tempfile.TemporaryDirectory() as tmpdir:
        # Датасет: температура vs продажи мороженого
        df = pd.DataFrame({
            "temperature_celsius": [15, 18, 22, 25, 28, 30, 33, 35, 38, 20,
                                     17, 29, 31, 24, 19, 27, 32, 26, 23, 21],
            "ice_cream_sales":     [100, 130, 180, 220, 290, 340, 400, 450, 510, 160,
                                     110, 310, 370, 210, 120, 270, 390, 250, 190, 150]
        })
        csv_path = os.path.join(tmpdir, "temperature_sales.csv")
        df.to_csv(csv_path, index=False)

        print(f"Датасет создан: {csv_path}")
        print(df.head())

        from popper import Popper

        agent = Popper(
            llm="deepseek/deepseek-v3.2",
            api_key=api_key
        )

        agent.register_data(
            data_path=tmpdir,
            loader_type='custom'
        )

        agent.configure(
            alpha=0.1,
            max_num_of_tests=3,
            max_retry=2,
            time_limit=1,
            relevance_checker=True,
            use_react_agent=True
        )

        hypothesis = (
            "Higher ambient temperature leads to significantly higher ice cream sales. "
            "Specifically, every 5°C increase in temperature corresponds to at least "
            "a 50-unit increase in daily sales."
        )

        print(f"\nГипотеза: {hypothesis}\n")
        print("Запускаю валидацию...")

        results = agent.validate(hypothesis=hypothesis)

        print("\n--- РЕЗУЛЬТАТЫ ---")
        print(f"Итог: {results['last_message']}")
        print(f"Разобранный результат: {results['parsed_result']}")


# -----------------------------------------------------------------------
# Раздел 3: Тест get_llm с OpenRouter (unit-тест)
# -----------------------------------------------------------------------
def test_get_llm_openrouter():
    """Проверяем, что get_llm правильно создаёт LLM для OpenRouter."""
    print("=" * 60)
    print("Тест 3: get_llm() с OpenRouter (unit-тест)")
    print("=" * 60)

    from popper.utils import get_llm

    llm = get_llm(model="deepseek/deepseek-v3.2", api_key="test-key-123")

    assert type(llm).__name__ == "CustomChatModel", f"Ожидался CustomChatModel, получен {type(llm).__name__}"
    assert llm.model_name == "deepseek/deepseek-v3.2"
    assert "openrouter.ai" in llm.openai_api_base

    print(f"✓ LLM тип: {type(llm).__name__}")
    print(f"✓ Модель: {llm.model_name}")
    print(f"✓ Base URL: {llm.openai_api_base}")
    print()


if __name__ == "__main__":
    print("POPPER + OpenRouter + Google Gemini — тесты\n")

    # Тест 3 не требует API ключа
    test_get_llm_openrouter()

    # Тесты 1 и 2 требуют OPENROUTER_API_KEY
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Для запуска тестов 1 и 2 задайте OPENROUTER_API_KEY:")
        print("  export OPENROUTER_API_KEY='your-key'")
        print("  python test_openrouter.py")
        sys.exit(0)

    test_openrouter_direct()
    test_popper_with_custom_data()

    print("Все тесты завершены.")
