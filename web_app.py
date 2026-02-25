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

DATA_GUIDELINES = """
### How to prepare your data

| Requirement | Details |
|---|---|
| **Format** | CSV files (comma or tab separated) |
| **Columns** | Use clear, descriptive English names (e.g. `education_years`, `annual_income_usd`) |
| **Rows** | At least 30+ rows for reliable statistical testing |
| **Data types** | Numbers and text — both work |
| **Multiple files** | You can upload several CSVs — each becomes a separate table |
| **Naming** | File name becomes table name: `my_data.csv` → `df_my_data` |

### Example

| respondent_id | education_years | annual_income_usd | gender | age |
|---|---|---|---|---|
| 1 | 14 | 71323 | male | 46 |
| 2 | 18 | 89500 | female | 32 |
| ... | ... | ... | ... | ... |

**Hypothesis example:** *"Higher levels of formal education are associated with significantly higher annual income."*
"""


def preview_files(files):
    """Show preview of uploaded CSV files."""
    if not files:
        return "No files uploaded yet."

    parts = []
    for f in files:
        fname = os.path.basename(f)
        try:
            df = pd.read_csv(f, nrows=5)
            table_name = f"df_{os.path.splitext(fname)[0]}"
            header = f"**{fname}** → `{table_name}` ({pd.read_csv(f).shape[0]} rows, {df.shape[1]} columns)\n"
            parts.append(header + df.to_markdown(index=False))
        except Exception as e:
            parts.append(f"**{fname}** — error: {e}")

    return "\n\n---\n\n".join(parts)


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
        yield [], "Upload at least one CSV file."
        return
    if not hypothesis.strip():
        yield [], "Enter a hypothesis."
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        yield [], "Set OPENROUTER_API_KEY in .env file."
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


with gr.Blocks(title="POPPER — Hypothesis Validator") as demo:
    gr.Markdown("# POPPER — Hypothesis Validator")
    gr.Markdown("Upload your data, enter a hypothesis, and let the AI agent test it using statistical falsification.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(DATA_GUIDELINES)

            files_input = gr.File(
                label="Upload CSV files",
                file_count="multiple",
                file_types=[".csv"],
            )
            file_preview = gr.Markdown("No files uploaded yet.")

            hypothesis_input = gr.Textbox(
                label="Hypothesis",
                placeholder="e.g. Higher education is associated with higher income...",
                lines=3,
            )

            domain_input = gr.Dropdown(
                choices=["sociology", "biology", "economics", "psychology", "political science", "education", "public health"],
                value="sociology",
                label="Domain",
            )

            with gr.Row():
                alpha_input = gr.Slider(
                    minimum=0.01, maximum=0.2, value=0.1, step=0.01,
                    label="Significance level (alpha)",
                )
                max_tests_input = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Max falsification tests",
                )

            validate_btn = gr.Button("Validate", variant="primary", size="lg")

        with gr.Column(scale=1):
            verdict_output = gr.Markdown("", label="Verdict")
            agent_log = gr.Chatbot(
                label="Agent Log",
                height=500,
            )

    # Events
    files_input.change(preview_files, inputs=[files_input], outputs=[file_preview])

    validate_btn.click(
        validate_hypothesis,
        inputs=[hypothesis_input, files_input, alpha_input, max_tests_input, domain_input],
        outputs=[agent_log, verdict_output],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
