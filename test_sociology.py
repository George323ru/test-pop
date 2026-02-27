"""
POPPER — тестирование социологических гипотез
через OpenRouter + Google Gemini

Использование:
    source venv/bin/activate
    python3 test_sociology.py

Как это работает:
    1. Вы кладёте CSV-файлы с данными в папку data/
    2. Задаёте гипотезу в свободной форме
    3. POPPER-агент (Gemini) сам придумывает статистические тесты
       для фальсификации гипотезы по вашим данным
    4. Получаете вердикт: подтверждена / фальсифицирована
"""

import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "sociology")


def create_sociology_datasets():
    """
    Создаём реалистичные социологические датасеты.
    В реальном применении — замените своими данными.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    np.random.seed(42)
    n = 200

    # --- Датасет 1: Образование и доход ---
    education_years = np.random.randint(8, 22, n)
    income = (
        15000
        + education_years * 3500
        + np.random.normal(0, 8000, n)
    ).clip(5000)
    df_education = pd.DataFrame({
        "respondent_id": range(1, n + 1),
        "education_years": education_years,        # лет обучения
        "annual_income_usd": income.astype(int),   # годовой доход
        "gender": np.random.choice(["male", "female"], n),
        "age": np.random.randint(22, 65, n),
        "region": np.random.choice(["urban", "suburban", "rural"], n, p=[0.5, 0.3, 0.2]),
    })
    path1 = os.path.join(DATA_DIR, "education_income.csv")
    df_education.to_csv(path1, index=False)
    print(f"Создан: {path1} ({len(df_education)} строк)")

    # --- Датасет 2: Социальное доверие и участие в выборах ---
    social_trust = np.random.uniform(0, 10, n)          # индекс доверия 0–10
    civic_engagement = np.random.uniform(0, 10, n)      # гражданская активность 0–10
    # Доверие → выше явка (с шумом)
    voter_turnout = (
        0.3 + 0.05 * social_trust + 0.03 * civic_engagement
        + np.random.normal(0, 0.08, n)
    ).clip(0.1, 1.0)
    df_social = pd.DataFrame({
        "respondent_id": range(1, n + 1),
        "social_trust_index": social_trust.round(2),
        "civic_engagement_score": civic_engagement.round(2),
        "voter_turnout_rate": voter_turnout.round(3),
        "income_bracket": np.random.choice(["low", "middle", "high"], n, p=[0.3, 0.5, 0.2]),
        "has_higher_education": (education_years > 15).astype(int),
    })
    path2 = os.path.join(DATA_DIR, "social_trust_voting.csv")
    df_social.to_csv(path2, index=False)
    print(f"Создан: {path2} ({len(df_social)} строк)")

    # --- Датасет 3: Уровень преступности и безработица ---
    unemployment_rate = np.random.uniform(3, 25, n)     # % безработица
    poverty_rate = unemployment_rate * 1.5 + np.random.normal(0, 5, n)
    # Безработица коррелирует с преступностью
    crime_rate = (
        20 + 2.5 * unemployment_rate + 1.2 * poverty_rate
        + np.random.normal(0, 15, n)
    ).clip(0)
    df_crime = pd.DataFrame({
        "district_id": range(1, n + 1),
        "unemployment_rate_pct": unemployment_rate.round(1),
        "poverty_rate_pct": poverty_rate.clip(0).round(1),
        "crime_rate_per_10k": crime_rate.round(1),           # преступлений на 10 000 жителей
        "police_per_10k": np.random.uniform(15, 50, n).round(1),
        "median_age": np.random.uniform(25, 50, n).round(1),
        "urbanization_pct": np.random.uniform(20, 100, n).round(1),
    })
    path3 = os.path.join(DATA_DIR, "crime_unemployment.csv")
    df_crime.to_csv(path3, index=False)
    print(f"Создан: {path3} ({len(df_crime)} строк)")

    print()
    return df_education, df_social, df_crime


def run_popper(hypothesis: str, description: str):
    """Запускает POPPER для проверки социологической гипотезы."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[ОШИБКА] OPENROUTER_API_KEY не задан в .env")
        sys.exit(1)

    from popper import Popper

    print("=" * 65)
    print(f"Гипотеза: {description}")
    print("=" * 65)
    print(f"Текст: {hypothesis}")
    print()

    agent = Popper(
        llm="deepseek/deepseek-v3.2",
        api_key=api_key,
    )

    agent.register_data(
        data_path=DATA_DIR,
        loader_type="custom",
    )

    agent.configure(
        alpha=0.1,            # порог значимости
        max_num_of_tests=3,   # количество фальсификационных тестов
        max_retry=2,
        time_limit=1,
        relevance_checker=True,
        use_react_agent=True,
    )

    results = agent.validate(hypothesis=hypothesis)

    print("\n--- ИТОГ ---")
    print(results["last_message"])
    if results["parsed_result"]:
        print(f"Структурированный ответ: {results['parsed_result']}")
    print()
    return results


if __name__ == "__main__":
    print("=" * 65)
    print("POPPER: Валидация социологических гипотез")
    print("Модель: deepseek/deepseek-v3.2 (OpenRouter)")
    print("=" * 65)
    print()

    # Шаг 1: Создаём датасеты (или используем существующие)
    print(">>> Подготовка данных...")
    create_sociology_datasets()

    # ---------------------------------------------------------------
    # Гипотеза 1: Образование → доход
    # Ожидаем: ПОДТВЕРЖДЕНА (данные специально так построены)
    # ---------------------------------------------------------------
    run_popper(
        hypothesis=(
            "Higher levels of formal education are associated with significantly "
            "higher annual income. Individuals with more years of schooling earn "
            "on average at least 3000 USD more per additional year of education."
        ),
        description="Образование → доход",
    )

    # ---------------------------------------------------------------
    # Гипотеза 2: Социальное доверие → явка на выборы
    # ---------------------------------------------------------------
    run_popper(
        hypothesis=(
            "Higher social trust is positively associated with higher voter turnout. "
            "Communities with greater interpersonal trust show meaningfully higher "
            "electoral participation rates."
        ),
        description="Социальное доверие → явка на выборах",
    )

    # ---------------------------------------------------------------
    # Гипотеза 3: Безработица → преступность
    # ---------------------------------------------------------------
    run_popper(
        hypothesis=(
            "Higher unemployment rates are positively associated with higher crime rates. "
            "Districts with unemployment above 15% have significantly more crimes "
            "per 10,000 residents than districts with unemployment below 10%."
        ),
        description="Безработица → преступность",
    )

    print("Все тесты завершены.")
