"""The question-answering workflow (LangGraph).

    classify -> generate_sql -> validate -> execute -> write_answer
                    ^______________|___________|
                       one retry with the error message

The model never touches the database: its SQL goes through the validator and the
read-only executor. The answer may only quote numbers that are in the rows.
"""

from __future__ import annotations

import json
import operator
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .grounding import allowed_numbers, ungrounded_numbers
from .llm import LLM, LLMError
from .schema_prompt import ANSWER_SYSTEM, CLASSIFY_SYSTEM, GENERATE_SYSTEM
from .sql_executor import QueryExecutionError, execute_readonly
from .sql_safety import MAX_ROWS, validate_sql

MAX_QUESTION_CHARS = 500
MAX_ATTEMPTS = 2  # first try + ONE retry
ROWS_SHOWN_TO_MODEL = 50

# Statuses: ok, no_data, needs_clarification, out_of_scope, suspicious,
#           unverified (answer failed the number check), error
SUSPICIOUS_REPLY = (
    "I can't help with that request. I answer read-only analytics questions about "
    "the retail sales and returns data."
)


@dataclass
class AgentResult:
    status: str
    answer: str
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    truncated: bool = False
    attempts: int = 0
    category: str | None = None
    warnings: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)  # [{"step": ..., "seconds": ...}]
    total_seconds: float = 0.0
    # [{"attempt": 1, "stage": "validate"|"execute", "ok": bool, "error": str}]
    attempt_log: list[dict] = field(default_factory=list)
    answer_rewrites: int = 0  # 1 if the first answer draft failed the number check
    error_kind: str | None = None  # LLMError.kind when the model call failed

    def to_dict(self) -> dict:
        return asdict(self)


class State(TypedDict, total=False):
    question: str
    category: str
    reason: str
    clarifying_question: str
    attempt: int
    sql: str  # last SQL the model wrote
    sql_to_run: str  # validated SQL (row limit enforced)
    error: str  # last validation / execution error (safe text)
    columns: list[str]
    rows: list[list]
    truncated: bool
    status: str
    answer: str
    warnings: list[str]
    steps: Annotated[list[dict], operator.add]
    attempt_log: Annotated[list[dict], operator.add]
    answer_rewrites: int
    error_kind: str


def _clean_question(question: str) -> str:
    """Remove anything that could close our delimiter tag early."""
    return re.sub(r"</?\s*user_question\s*>", " ", question, flags=re.IGNORECASE).strip()


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _extract_sql(text: str) -> str:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def metric_definition(sql: str) -> str | None:
    """Which revenue definition a query used, decided from the SQL itself."""
    uses_sales = re.search(r"\brevenue\b", sql, re.IGNORECASE) is not None
    uses_returns = re.search(r"\breturn_value\b", sql, re.IGNORECASE) is not None
    if uses_sales and uses_returns:
        return (
            "Definition used: net revenue = sales revenue (quantity x unit price) "
            "minus the value of returns, in GBP; returns count in the period they were issued."
        )
    if uses_sales:
        return (
            "Definition used: gross sales revenue = quantity x unit price, in GBP, "
            "before subtracting returns."
        )
    if uses_returns:
        return "Definition used: return value = units returned x unit price, in GBP."
    return None


PARTIAL_MONTH_NOTE = "Note: the data ends on 2011-12-09, so December 2011 covers only 1-9 December."


KNOWN_CUSTOMERS_NOTE = (
    "Note: customer figures only include sales lines with a known customer ID; "
    "lines from guest or unknown buyers are excluded."
)
GUEST_ONLY_NOTE = "Note: this covers only sales lines without a customer ID (guest or unknown buyers)."
_KNOWN = re.compile(r"\bhas_customer_id\s*(?:=\s*1\b|(?:<>|!=)\s*0\b|>\s*0\b|in\s*\(\s*1\s*\))", re.IGNORECASE)
_GUEST = re.compile(r"\bhas_customer_id\s*(?:=\s*0\b|(?:<>|!=)\s*1\b|<\s*1\b|in\s*\(\s*0\s*\))", re.IGNORECASE)


def customer_scope_note(sql: str) -> str | None:
    """A caveat when the executed SQL filters on has_customer_id. A query that
    compares against both known and unknown (e.g. a known/guest split) gets none."""
    known, guest = bool(_KNOWN.search(sql)), bool(_GUEST.search(sql))
    if known and not guest:
        return KNOWN_CUSTOMERS_NOTE
    if guest and not known:
        return GUEST_ONLY_NOTE
    return None


def _mentions_december_2011(rows: list[list]) -> bool:
    return any(isinstance(c, str) and c.startswith("2011-12") for row in rows for c in row)


def _fallback_answer(columns: list[str], rows: list[list]) -> str:
    lines = ["I could not verify a written summary against the data, so here are the returned rows:"]
    lines.append(" | ".join(columns))
    lines.extend(" | ".join(str(c) for c in row) for row in rows[:10])
    if len(rows) > 10:
        lines.append(f"... and {len(rows) - 10} more rows.")
    return "\n".join(lines)


class RetailAgent:
    def __init__(
        self,
        llm: LLM,
        db_path: str | Path,
        *,
        max_rows: int = MAX_ROWS,
        query_timeout_s: float = 5.0,
    ):
        self.llm = llm
        self.db_path = Path(db_path)
        self.max_rows = max_rows
        self.query_timeout_s = query_timeout_s
        self.graph = self._build_graph()

    # -- helpers ---------------------------------------------------------
    def _timed(self, name: str, fn: Callable[[State], dict]) -> Callable[[State], dict]:
        def node(state: State) -> dict:
            started = time.perf_counter()
            try:
                update = fn(state)
            except LLMError as error:  # provider problem: stop with a clear message
                update = {"status": "error", "answer": str(error), "error_kind": error.kind}
            seconds = round(time.perf_counter() - started, 3)
            return {**update, "steps": [{"step": name, "seconds": seconds}]}

        return node

    # -- nodes -----------------------------------------------------------
    def _classify(self, state: State) -> dict:
        user = f"<user_question>\n{state['question']}\n</user_question>"
        reply = _extract_json(self.llm.complete(CLASSIFY_SYSTEM, user, step="classify", max_tokens=600)) or {}
        category = str(reply.get("category", "")).strip().lower()
        if category not in {"answerable", "needs_clarification", "out_of_scope", "suspicious"}:
            category = "answerable"  # not a security boundary: validator + executor still guard
        update: dict[str, Any] = {
            "category": category,
            "reason": str(reply.get("reason", ""))[:200],
            "clarifying_question": str(reply.get("clarifying_question", ""))[:300],
        }
        if category == "needs_clarification":
            update.update(
                status="needs_clarification",
                answer=update["clarifying_question"]
                or "Could you tell me more precisely what you would like to know?",
            )
        elif category == "out_of_scope":
            update.update(
                status="out_of_scope",
                answer="That is outside what I can answer from the retail sales and returns data. "
                + update["reason"],
            )
        elif category == "suspicious":
            update.update(status="suspicious", answer=SUSPICIOUS_REPLY)
        return update

    def _generate(self, state: State) -> dict:
        user = f"<user_question>\n{state['question']}\n</user_question>"
        if state.get("error"):
            user += (
                f"\n\nYour previous SQL was rejected.\nPrevious SQL:\n{state.get('sql', '')}\n"
                f"Error: {state['error']}\nWrite a corrected single SELECT."
            )
        text = self.llm.complete(GENERATE_SYSTEM, user, step="generate_sql", max_tokens=1500)
        attempt = state.get("attempt", 0) + 1
        if text.upper().startswith("CANNOT_ANSWER"):
            reason = text.split(":", 1)[1].strip()[:200] if ":" in text else ""
            return {
                "attempt": attempt,
                "status": "out_of_scope",
                "answer": "The data cannot answer that. " + reason,
            }
        return {"attempt": attempt, "sql": _extract_sql(text), "error": ""}

    def _validate(self, state: State) -> dict:
        result = validate_sql(state["sql"], self.max_rows)
        log = {"attempt": state.get("attempt", 0), "stage": "validate", "ok": result.ok, "error": result.error or ""}
        if result.ok:
            return {"sql_to_run": result.sql, "error": "", "attempt_log": [log]}
        return {"error": f"Validation failed: {result.error}", "attempt_log": [log]}

    def _execute(self, state: State) -> dict:
        try:
            result = execute_readonly(
                self.db_path, state["sql_to_run"], self.max_rows, self.query_timeout_s
            )
        except QueryExecutionError as error:
            log = {"attempt": state.get("attempt", 0), "stage": "execute", "ok": False, "error": str(error)}
            return {"error": f"Execution failed: {error}", "attempt_log": [log]}
        log = {"attempt": state.get("attempt", 0), "stage": "execute", "ok": True, "error": ""}
        return {
            "attempt_log": [log],
            "error": "",
            "columns": result.columns,
            "rows": [list(r) for r in result.rows],
            "truncated": result.truncated,
        }

    def _give_up(self, state: State) -> dict:
        return {
            "status": "error",
            "answer": (
                "I could not produce a valid, safe query for that question, so I will not guess "
                f"an answer. Last problem: {state.get('error', 'unknown')}"
            ),
        }

    def _write_answer(self, state: State) -> dict:
        columns, rows = state["columns"], state["rows"]
        if not rows:
            return {
                "status": "no_data",
                "answer": "The query ran successfully but returned no rows, so there is no data "
                "for that question.",
            }
        allowed = allowed_numbers(state["question"], columns, rows)
        shown = rows[:ROWS_SHOWN_TO_MODEL]
        user = (
            f"<user_question>\n{state['question']}\n</user_question>\n\n"
            f"SQL that was run:\n{state['sql_to_run']}\n\n"
            f"<query_results>\ncolumns: {json.dumps(columns)}\n"
            f"rows ({len(shown)} of {len(rows)} shown, truncated={state['truncated']}):\n"
            f"{json.dumps(shown, default=str)}\n</query_results>"
        )
        text = self.llm.complete(ANSWER_SYSTEM, user, step="write_answer", max_tokens=1200)
        bad = ungrounded_numbers(text, allowed)
        rewrites = 1 if bad else 0
        if bad:  # one rewrite, naming the offending numbers
            text = self.llm.complete(
                ANSWER_SYSTEM,
                user + f"\n\nYour previous answer used numbers not in the rows: {bad}. "
                "Rewrite it using only numbers from the rows.",
                step="write_answer",
                max_tokens=1200,
            )
            bad = ungrounded_numbers(text, allowed)
        warnings = list(state.get("warnings", []))
        if bad:
            warnings.append(f"Answer quoted numbers that are not in the rows: {bad}")
            return {
                "status": "unverified",
                "answer": _fallback_answer(columns, rows),
                "warnings": warnings,
                "answer_rewrites": rewrites,
            }
        definition = metric_definition(state["sql_to_run"])
        if state["truncated"]:
            text += f"\n(Only the first {len(rows)} rows are shown; the result was cut off.)"
        if _mentions_december_2011(rows):  # added by code: the model got this wrong once
            text += f"\n{PARTIAL_MONTH_NOTE}"
        scope_note = customer_scope_note(state["sql_to_run"])  # added by code: the model omitted it
        if scope_note:
            text += f"\n{scope_note}"
        if definition:
            text += f"\n{definition}"
        return {"status": "ok", "answer": text, "warnings": warnings, "answer_rewrites": rewrites}

    # -- routing ---------------------------------------------------------
    @staticmethod
    def _after_classify(state: State) -> str:
        if state.get("status"):
            return "end"
        return "generate_sql"

    @staticmethod
    def _after_generate(state: State) -> str:
        return "end" if state.get("status") else "validate"

    @staticmethod
    def _after_validate(state: State) -> str:
        if not state.get("error"):
            return "execute"
        return "generate_sql" if state.get("attempt", 0) < MAX_ATTEMPTS else "give_up"

    @staticmethod
    def _after_execute(state: State) -> str:
        if not state.get("error"):
            return "write_answer"
        return "generate_sql" if state.get("attempt", 0) < MAX_ATTEMPTS else "give_up"

    def _build_graph(self):
        graph = StateGraph(State)
        for name, fn in {
            "classify": self._classify,
            "generate_sql": self._generate,
            "validate": self._validate,
            "execute": self._execute,
            "write_answer": self._write_answer,
            "give_up": self._give_up,
        }.items():
            graph.add_node(name, self._timed(name, fn))
        graph.add_edge(START, "classify")
        graph.add_conditional_edges("classify", self._after_classify, {"end": END, "generate_sql": "generate_sql"})
        graph.add_conditional_edges("generate_sql", self._after_generate, {"end": END, "validate": "validate"})
        graph.add_conditional_edges(
            "validate", self._after_validate, {"execute": "execute", "generate_sql": "generate_sql", "give_up": "give_up"}
        )
        graph.add_conditional_edges(
            "execute", self._after_execute,
            {"write_answer": "write_answer", "generate_sql": "generate_sql", "give_up": "give_up"},
        )
        graph.add_edge("write_answer", END)
        graph.add_edge("give_up", END)
        return graph.compile()

    # -- public ----------------------------------------------------------
    def ask(self, question: str) -> AgentResult:
        started = time.perf_counter()
        question = _clean_question(question or "")
        if not question:
            return AgentResult(status="needs_clarification", answer="Please type a question about the retail data.")
        if len(question) > MAX_QUESTION_CHARS:
            return AgentResult(
                status="needs_clarification",
                answer=f"Please keep the question under {MAX_QUESTION_CHARS} characters.",
            )
        final = self.graph.invoke(
            {"question": question, "attempt": 0, "steps": [], "warnings": [], "attempt_log": []}
        )
        status = final.get("status") or "error"
        succeeded = status in {"ok", "unverified", "no_data"}
        return AgentResult(
            status=status,
            answer=final.get("answer", ""),
            sql=final.get("sql_to_run") if succeeded else final.get("sql"),
            columns=final.get("columns", []),
            rows=final.get("rows", []),
            truncated=final.get("truncated", False),
            attempts=final.get("attempt", 0),
            category=final.get("category"),
            warnings=final.get("warnings", []),
            steps=final.get("steps", []),
            total_seconds=round(time.perf_counter() - started, 3),
            attempt_log=final.get("attempt_log", []),
            answer_rewrites=final.get("answer_rewrites", 0),
            error_kind=final.get("error_kind"),
        )
