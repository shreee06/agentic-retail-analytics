"""Smoke tests for the Streamlit app with a MOCKED language model (no key, no network)."""

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.retail_analytics.llm import GroqLLM, LLMError

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
SECRET = "gsk_THIS_MUST_NEVER_BE_SHOWN_123456"


class ScriptedLLM:
    """Fake model: replies per step; ``category`` drives the classifier."""

    model = "fake/model"

    def __init__(self, category="answerable", sql="SELECT COUNT(*) AS n FROM returns",
                 answer="There is 1 return line.", error=None, clarify="Which period?"):
        self.category, self.sql, self.answer, self.error, self.clarify = category, sql, answer, error, clarify
        self.questions = []

    def complete(self, system, user, *, step="", max_tokens=1500):
        if self.error:
            raise self.error
        match = re.search(r"<user_question>\n(.*?)\n</user_question>", user, re.S)
        if match:
            self.questions.append(match.group(1))
        if step == "classify":
            return json.dumps({"category": self.category, "reason": "because", "clarifying_question": self.clarify})
        if step == "generate_sql":
            return self.sql
        return self.answer


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch, retail_db):
    st.cache_resource.clear()
    monkeypatch.setenv("DB_PATH", str(retail_db))
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    yield
    st.cache_resource.clear()


def launch(llm=None, from_env_error=None):
    """Start the app with GroqLLM.from_env replaced by the fake."""
    def fake_from_env(*a, **k):
        if from_env_error:
            raise from_env_error
        return llm
    patcher = patch.object(GroqLLM, "from_env", staticmethod(fake_from_env))
    patcher.start()
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at, patcher


def ask(at, text):
    at.text_input(key="question_text").set_value(text)
    at.button(key="ask").click().run()
    return at


def all_text(at) -> str:
    parts = []
    for kind in ("markdown", "caption", "error", "warning", "info", "code", "text", "title"):
        parts += [str(e.value) for e in getattr(at, kind)]
    parts += [str(e.label) for e in at.expander] + [str(e.label) for e in at.button]
    for df in at.dataframe:
        parts.append(df.value.to_string())
    return "\n".join(parts)


def badge(at) -> str:
    return next(m.value for m in at.markdown if "-badge[" in m.value)


def test_first_render_has_question_box_samples_and_limitations_but_no_sql_box():
    at, patcher = launch(ScriptedLLM())
    try:
        assert not at.exception
        assert len(at.text_input) == 1 and at.text_input[0].label == "Your question"
        assert len(at.text_area) == 0  # no free SQL input anywhere
        assert not [t for t in at.text_input if "sql" in t.label.lower()]
        assert sorted(b.key for b in at.button if b.key.startswith("sample_")) == [f"sample_{i}" for i in range(6)]
        assert at.button(key="ask") is not None
        assert "Limitations" in [e.label for e in at.expander]
        assert not at.error and at.get("code") == []
        assert "Model: fake/model" in all_text(at)
    finally:
        patcher.stop()


def test_ok_answer_shows_badge_answer_definition_sql_table_and_timings():
    llm = ScriptedLLM(sql="SELECT ROUND(SUM(revenue), 2) AS revenue FROM sales WHERE country IS NOT NULL",
                      answer="Revenue is 57.77 GBP.")
    at, patcher = launch(llm)
    try:
        ask(at, "What is the revenue?")
        assert not at.exception
        assert ":green-badge[OK]" in badge(at)
        text = all_text(at)
        assert "Revenue is 57.77 GBP." in text and "What is the revenue?" in text
        assert any("Definition used: gross sales revenue" in c.value for c in at.caption)
        assert "SQL used" in [e.label for e in at.expander] and "Timings" in [e.label for e in at.expander]
        assert "SELECT" in at.code[0].value and at.code[0].language == "sql"
        assert at.dataframe[0].value.iloc[0, 0] == 57.77
        assert "Total" in text and "SQL attempt" in text
        assert not at.error and not at.warning
    finally:
        patcher.stop()


def test_sample_button_runs_that_question():
    llm = ScriptedLLM()
    at, patcher = launch(llm)
    try:
        at.button(key="sample_4").click().run()
        assert llm.questions and llm.questions[0] == "How many customers placed more than 5 orders?"
        assert at.text_input(key="question_text").value == llm.questions[0]
        assert ":green-badge[OK]" in badge(at)
    finally:
        patcher.stop()


@pytest.mark.parametrize("kwargs,label,colour", [
    (dict(category="needs_clarification", clarify="Which period do you mean?"), "Needs clarification", "blue"),
    (dict(category="out_of_scope"), "Out of scope", "orange"),
    (dict(category="suspicious"), "Suspicious", "red"),
])
def test_refusal_badges(kwargs, label, colour):
    at, patcher = launch(ScriptedLLM(**kwargs))
    try:
        ask(at, "Something")
        assert f":{colour}-badge[{label}]" in badge(at)
        assert not at.exception and at.get("code") == []  # no SQL was written
        assert "SQL used" not in [e.label for e in at.expander]
        assert [e.label for e in at.expander if e.label == "Timings"] == ["Timings"]
        assert len(at.dataframe) == 1  # only the timings table, no result rows
    finally:
        patcher.stop()


def test_clarifying_question_is_shown():
    at, patcher = launch(ScriptedLLM(category="needs_clarification", clarify="Which period do you mean?"))
    try:
        ask(at, "Compare it")
        assert "Which period do you mean?" in all_text(at)
    finally:
        patcher.stop()


def test_unverified_badge_and_warning_when_the_answer_has_unsupported_numbers():
    at, patcher = launch(ScriptedLLM(answer="There are 987,654 return lines."))
    try:
        ask(at, "How many returns?")
        assert ":orange-badge[Unverified]" in badge(at)
        assert any("not in the rows" in w.value for w in at.warning)
        assert "987,654" not in "".join(m.value for m in at.markdown if "-badge[" not in m.value)
        assert len(at.dataframe) >= 1
    finally:
        patcher.stop()


def test_error_badge_when_the_sql_is_rejected_twice():
    at, patcher = launch(ScriptedLLM(sql="DROP TABLE sales"))
    try:
        ask(at, "Do something odd")
        assert ":red-badge[Error]" in badge(at)
        assert any("will not guess" in e.value for e in at.error)
        assert not at.exception
    finally:
        patcher.stop()


def test_truncated_results_show_a_warning():
    at, patcher = launch(ScriptedLLM(sql="SELECT * FROM sales"))
    try:
        ask(at, "Show all sales")
        assert any("cut off" in w.value for w in at.warning)
        assert len(at.dataframe[0].value) == 1000
    finally:
        patcher.stop()


def test_groq_outage_is_a_short_clear_error_without_details():
    outage = LLMError("Groq is temporarily unavailable. Please try again later.", "server")
    at, patcher = launch(ScriptedLLM(error=outage))
    try:
        ask(at, "Anything")
        assert ":red-badge[Error]" in badge(at)
        assert any("temporarily unavailable" in e.value for e in at.error)
        assert "Traceback" not in all_text(at)
    finally:
        patcher.stop()


def test_missing_key_shows_a_clear_message_and_disables_input():
    at, patcher = launch(from_env_error=LLMError("GROQ_API_KEY is not set. Add it to the .env file (see .env.example)."))
    try:
        assert not at.exception
        assert any("GROQ_API_KEY is not set" in e.value for e in at.error)
        assert at.text_input(key="question_text").disabled and at.button(key="ask").disabled
        assert all(b.disabled for b in at.button if b.key.startswith("sample_"))
        assert "Limitations" in [e.label for e in at.expander]  # the page still renders
    finally:
        patcher.stop()


def test_missing_database_is_reported_without_a_path(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "secret_folder" / "nope.sqlite"))
    at, patcher = launch(ScriptedLLM())
    try:
        assert any("database file is missing" in e.value for e in at.error)
        assert "secret_folder" not in all_text(at)
        assert at.button(key="ask").disabled
    finally:
        patcher.stop()


def test_unexpected_crash_in_the_agent_shows_no_internals():
    llm = ScriptedLLM()
    llm.complete = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom /home/someone/secret.py"))
    at, patcher = launch(llm)
    try:
        ask(at, "Anything")
        text = all_text(at)
        assert "boom" not in text and "/home/" not in text and "Traceback" not in text
        assert not at.exception
    finally:
        patcher.stop()


def test_empty_question_asks_for_input():
    at, patcher = launch(ScriptedLLM())
    try:
        ask(at, "   ")
        assert any("Type a question first" in i.value for i in at.info)
    finally:
        patcher.stop()


def test_the_key_is_never_displayed_anywhere():
    llm = ScriptedLLM()
    at, patcher = launch(llm)
    try:
        ask(at, "What is the revenue?")
        assert SECRET not in all_text(at)
        assert SECRET not in json.dumps([str(e) for e in at])  # nothing in the element tree either
    finally:
        patcher.stop()


def test_answer_text_cannot_inject_html_or_maths():
    at, patcher = launch(ScriptedLLM(answer="1 return line. <script>alert(1)</script> costs $1 and $1"))
    try:
        ask(at, "How many returns?")
        md = next(m for m in at.markdown if "<script>" in m.value)
        assert md.proto.allow_html is False and "\\$1" in md.value
    finally:
        patcher.stop()
