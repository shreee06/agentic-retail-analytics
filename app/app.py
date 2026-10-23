"""Streamlit demo: ask a question, get a validated, read-only, grounded answer.

Run: env -u PYTHONPATH .venv/bin/streamlit run app/app.py
There is deliberately no SQL input box: the model writes the SQL, the validator and
the read-only executor check it, and the page only shows what was run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.retail_analytics.agent import MAX_QUESTION_CHARS, RetailAgent  # noqa: E402
from src.retail_analytics.llm import GroqLLM, LLMError  # noqa: E402
from src.retail_analytics.pacing import PacedLLM  # noqa: E402

SAMPLE_QUESTIONS = [
    "What is the total net revenue?",
    "Show the net revenue for each month.",
    "What are the 10 best-selling products by net revenue? Give the stock codes.",
    "Which country has the highest return value?",
    "How many customers placed more than 5 orders?",
    "What was our profit margin last month?",
]

# status -> (badge label, colour, one-line meaning)
BADGES = {
    "ok": ("OK", "green", "The query ran and the answer was checked against the rows."),
    "no_data": ("No data", "gray", "The query ran but returned no rows."),
    "needs_clarification": ("Needs clarification", "blue", "The question is too vague to answer without guessing."),
    "out_of_scope": ("Out of scope", "orange", "The data cannot answer this question."),
    "suspicious": ("Suspicious", "red", "The request was refused (it tried to change data, see internals or bypass the rules)."),
    "unverified": ("Unverified", "orange", "The written summary failed the number check, so the raw rows are shown instead."),
    "error": ("Error", "red", "No answer was produced; nothing was guessed."),
}
NOTE_PREFIXES = ("Definition used:", "Note:", "(Only the first")

LIMITATIONS = """
* **Revenue** means *net revenue* (sales minus returns) unless you ask for gross. The definition used is shown under each answer.
* **Customer questions** count only sales lines with a known customer ID; about a quarter of sales lines have none.
* **Data** covers 2010-12-01 to 2011-12-09, so December 2011 is a partial month. Returns count in the period they were issued.
* **Two huge orders** (invoices 541431 and 581483) were cancelled the same day. They inflate *gross* revenue and "units sold" rankings.
* **A product** is a stock code. Some stock codes carry two descriptions, so asking for products by name can give different numbers.
* **Checks cover digits, not words.** Every number in an answer must come from the returned rows, but a sentence can still be misleading. In a manual review of 12 answers, 2 omitted a caveat.
* The assistant can **misjudge a vague question** (it may try to answer instead of asking). Queries run read-only, return at most 1,000 rows and stop after 5 seconds.
* It is a demo on a public dataset, evaluated on 60 questions; see the README for the results and limits.
"""


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@st.cache_resource(show_spinner=False)
def load_agent(db_path: str) -> tuple[RetailAgent, str]:
    inner = GroqLLM.from_env()  # raises LLMError when the key is missing; never cached
    llm = PacedLLM(inner, min_interval_s=0.0, max_retries=2, base_backoff_s=3.0)
    return RetailAgent(llm, db_path), inner.model


def safe_markdown(text: str) -> str:
    return text.replace("$", "\\$")  # keep "$" from being read as maths


def render_result(question: str, result) -> None:
    label, colour, meaning = BADGES.get(result.status, BADGES["error"])
    st.markdown(f":{colour}-badge[{label}]  {meaning}")

    lines = result.answer.splitlines()
    notes = [line for line in lines if line.startswith(NOTE_PREFIXES)]
    body = "\n".join(line for line in lines if not line.startswith(NOTE_PREFIXES)).strip()
    if result.status == "error":
        st.error(body or "Something went wrong.")
    else:
        st.markdown(safe_markdown(body))
    for note in notes:
        st.caption(safe_markdown(note))
    for warning in result.warnings:
        st.warning(warning)

    if result.sql:
        with st.expander("SQL used"):
            st.code(result.sql, language="sql")
            st.caption("Written by the model, then validated and run read-only exactly as shown.")

    if result.rows:
        frame = pd.DataFrame(result.rows, columns=result.columns)
        st.dataframe(frame, width="stretch", hide_index=True)
        if result.truncated:
            st.warning(f"Showing the first {len(result.rows):,} rows; the result was cut off.")

    if result.steps:
        with st.expander("Timings"):
            st.dataframe(
                pd.DataFrame(result.steps).rename(columns={"step": "Step", "seconds": "Seconds"}),
                width="stretch", hide_index=True,
            )
            st.caption(f"Total {result.total_seconds:.2f} s, {result.attempts} SQL attempt(s).")


def pick_sample(text: str) -> None:
    st.session_state["question_text"] = text
    st.session_state["run_requested"] = True


def request_run() -> None:
    st.session_state["run_requested"] = True


def main() -> None:
    st.set_page_config(page_title="Retail Analytics Assistant", page_icon="📊", layout="centered")
    st.title("Retail Analytics Assistant")
    st.caption("Ask about sales and returns of a UK online gift retailer (UCI Online Retail, Dec 2010 to Dec 2011).")

    db_path = project_path(os.environ.get("DB_PATH", "data/processed/retail.sqlite"))
    agent = model = None
    problem = None
    if not db_path.is_file():
        problem = "The database file is missing. Build it first: scripts/build_db.py (see the README)."
    else:
        try:
            agent, model = load_agent(str(db_path))
        except LLMError as error:
            problem = str(error)
        except Exception:
            problem = "The assistant could not be started."
    if problem:
        st.error(problem)

    st.text_input("Your question", key="question_text", max_chars=MAX_QUESTION_CHARS,
                  placeholder="e.g. What was the net revenue in March 2011?", disabled=agent is None,
                  on_change=request_run)
    st.button("Ask", key="ask", type="primary", disabled=agent is None, on_click=request_run)
    st.caption("Try one of these:")
    columns = st.columns(2)
    for i, sample in enumerate(SAMPLE_QUESTIONS):
        columns[i % 2].button(sample, key=f"sample_{i}", on_click=pick_sample, args=(sample,),
                              disabled=agent is None, width="stretch")

    if st.session_state.pop("run_requested", False) and agent is not None:
        question = st.session_state.get("question_text", "").strip()
        if question:
            with st.spinner("Thinking..."):
                try:
                    st.session_state["last"] = (question, agent.ask(question))
                except Exception:  # never show internals
                    st.session_state["last"] = None
                    st.error("Something went wrong while answering. Please try again.")
        else:
            st.info("Type a question first.")

    last = st.session_state.get("last")
    if last:
        st.divider()
        st.markdown(f"**Question:** {safe_markdown(last[0])}")
        render_result(*last)

    if model:
        st.caption(f"Model: {model}")
    with st.expander("Limitations"):
        st.markdown(LIMITATIONS)


main()
