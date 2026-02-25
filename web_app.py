"""
POPPER Web Interface
Запуск: python3 web_app.py
"""

import os
import shutil
import tempfile
import asyncio
import copy

import gradio as gr
try:
    from gradio import ChatMessage
except ImportError:
    ChatMessage = None
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from popper import Popper

MODEL = "google/gemini-3-flash-preview"

HOW_TO_USE = """
### Как пользоваться

1. **Загрузите данные** — один или несколько CSV-файлов (разделитель — запятая или табуляция, минимум 30 строк)
2. **Напишите гипотезу** — утверждение, которое вы хотите проверить на ваших данных
3. **Выберите домен** — область исследования (социология, экономика и т.д.)
4. **Нажмите «Проверить»** — AI-агент разработает статистические тесты, выполнит их и вынесет вердикт

**Пример гипотезы:** *«Мужчины оценивают честность значимо ниже, чем женщины»*
"""


async def validate_hypothesis(
    hypothesis,
    files,
    alpha,
    max_tests,
    domain,
    progress=gr.Progress(track_tqdm=True),
):
    """Run POPPER validation and stream agent logs."""
    if not files:
        yield [], "Загрузите хотя бы один CSV-файл."
        return
    if not hypothesis.strip():
        yield [], "Введите гипотезу."
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        yield [], "Не задан OPENROUTER_API_KEY. Обратитесь к администратору."
        return

    # Copy uploaded files to a temp directory
    tmpdir = tempfile.mkdtemp(prefix="popper_")
    try:
        for f in files:
            shutil.copy2(f, os.path.join(tmpdir, os.path.basename(f)))

        agent = Popper(llm=MODEL, api_key=api_key)
        agent.register_data(data_path=tmpdir, loader_type="custom")
        agent.configure(
            alpha=alpha,
            max_num_of_tests=int(max_tests),
            max_retry=3,
            time_limit=1,
            relevance_checker=True,
            use_react_agent=True,
            domain=domain,
        )

        prev_log = copy.deepcopy(agent.agent.log)

        task = asyncio.create_task(
            asyncio.to_thread(agent.agent.go, hypothesis)
        )

        def log_to_messages(log):
            msgs = []
            sections = [
                ("designer", "Experiment Designer"),
                ("executor", "Executor"),
                ("relevance_checker", "Relevance Checker"),
                ("sequential_testing", "Sequential Testing"),
                ("summarizer", "Summarizer"),
            ]
            for key, label in sections:
                for entry in log.get(key, []):
                    if ChatMessage is not None:
                        msgs.append(ChatMessage(role="assistant", content=f"**[{label}]**\n\n{entry}"))
                    else:
                        msgs.append({"role": "assistant", "content": f"**[{label}]**\n\n{entry}"})
            return msgs

        while not task.done():
            await asyncio.sleep(1)
            if agent.agent.log != prev_log:
                prev_log = copy.deepcopy(agent.agent.log)
                yield log_to_messages(prev_log), ""

        result = await task
        log, last_message, parsed_result = result

        final_msgs = log_to_messages(agent.agent.log)

        verdict = ""
        if parsed_result:
            conclusion = parsed_result.get("conclusion")
            verdict_icon = "supported" if conclusion else "falsified"
            verdict = f"### Result: Hypothesis **{verdict_icon}**\n\n"
            verdict += f"**Reasoning:** {parsed_result.get('reasoning', '')}\n\n"
            verdict += f"**Rationale:** {parsed_result.get('rationale', '')}"
        else:
            verdict = f"### Result\n\n{last_message}"

        yield final_msgs, verdict

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


with gr.Blocks(title="POPPER — Проверка гипотез") as demo:
    gr.Markdown("# POPPER — Проверка гипотез")
    gr.Markdown("Загрузите данные, введите гипотезу — AI-агент проверит её методом статистической фальсификации.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(HOW_TO_USE)

            files_input = gr.File(
                label="Загрузите CSV-файлы",
                file_count="multiple",
                file_types=[".csv"],
            )

            hypothesis_input = gr.Textbox(
                label="Гипотеза",
                placeholder="Например: Мужчины оценивают честность значимо ниже, чем женщины...",
                lines=3,
            )

            domain_input = gr.Dropdown(
                choices=["sociology", "biology", "economics", "psychology", "political science", "education", "public health"],
                value="sociology",
                label="Домен (область исследования)",
            )

            with gr.Row():
                alpha_input = gr.Slider(
                    minimum=0.01, maximum=0.2, value=0.1, step=0.01,
                    label="Уровень значимости (alpha)",
                    info="Порог ошибки: 0.05 — строгий, 0.1 — стандартный, 0.2 — мягкий",
                )
                max_tests_input = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Кол-во тестов",
                    info="Сколько статистических тестов проведёт агент (больше = надёжнее, но дольше)",
                )

            validate_btn = gr.Button("Проверить", variant="primary", size="lg")

        with gr.Column(scale=1):
            verdict_output = gr.Markdown("", label="Вердикт")

    # Agent Log — внизу, на всю ширину
    agent_log = gr.Chatbot(
        label="Лог агента (подробности работы)",
        height=500,
    )

    # Events
    validate_btn.click(
        validate_hypothesis,
        inputs=[hypothesis_input, files_input, alpha_input, max_tests_input, domain_input],
        outputs=[agent_log, verdict_output],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue(max_size=5)
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
