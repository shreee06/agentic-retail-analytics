"""Tests of the evaluation harness itself, using fake LLMs (no key, no network)."""

import csv
import json
import re
from pathlib import Path

import pytest

from src.retail_analytics import evaluation as ev
from src.retail_analytics.agent import RetailAgent, _clean_question
from src.retail_analytics.llm import LLMError
from src.retail_analytics.pacing import PacedLLM

EVALS = Path(__file__).resolve().parents[1] / "evals"
DEV = ev.load_questions(EVALS / "dev_set.json")
TEST = ev.load_questions(EVALS / "test_set.json")
USER = ev.load_questions(EVALS / "test_set_user.json")


class OracleLLM:
    """Plays a perfect assistant: classifies by the expected status, writes the gold SQL,
    and echoes the returned rows (so every number is grounded)."""

    CATEGORY = {"ok": "answerable", "needs_clarification": "needs_clarification",
                "out_of_scope": "out_of_scope", "suspicious": "suspicious"}

    def __init__(self, questions, sql_override=None):
        # the agent strips delimiter tags from questions, so look them up the same way
        self.by_question = {_clean_question(q["question"]): q for q in questions}
        self.sql_override = sql_override
        self.calls = 0

    def complete(self, system, user, *, step="", max_tokens=1500):
        self.calls += 1
        q = self.by_question[re.search(r"<user_question>\n(.*?)\n</user_question>", user, re.S).group(1)]
        if step == "classify":
            return json.dumps({"category": self.CATEGORY[q["expected_status"]], "reason": "r",
                               "clarifying_question": "Which period do you mean?"})
        if step == "generate_sql":
            return self.sql_override or q["gold_sql"]
        rows = json.loads(re.search(r"rows \(.*?\):\n(.*)\n</query_results>", user, re.S).group(1))
        return "Result: " + "; ".join(", ".join(str(c) for c in row) for row in rows)


class FlakyLLM:
    """Raises a rate-limit error for the first ``failures`` calls, then delegates."""

    def __init__(self, inner, failures, retry_after=None):
        self.inner, self.failures, self.retry_after = inner, failures, retry_after

    def complete(self, system, user, *, step="", max_tokens=1500):
        if self.failures > 0:
            self.failures -= 1
            raise LLMError("Groq's rate limit was reached.", "rate_limit", self.retry_after)
        return self.inner.complete(system, user, step=step, max_tokens=max_tokens)


def evaluate(llm, db, questions, paced=None):
    return ev.run_evaluation(RetailAgent(paced or llm, db), paced, questions, db)


# --- the question files -----------------------------------------------------
def test_question_sets_have_the_planned_size_and_categories():
    assert 12 <= len(DEV) <= 18 and 38 <= len(TEST) <= 45
    wanted = {"simple_aggregation", "filtering", "customer_level", "returns_net", "period_comparison",
              "ambiguous", "unanswerable", "prompt_injection", "modify_data"}
    assert {q["category"] for q in TEST} == wanted
    assert {q["category"] for q in DEV} == wanted


def test_ids_are_unique_and_dev_and_test_do_not_overlap():
    ids = [q["id"] for q in DEV + TEST + USER]
    assert len(ids) == len(set(ids))
    assert not {q["question"].lower() for q in DEV} & {q["question"].lower() for q in TEST}


def test_every_answerable_question_has_a_validator_approved_gold_query_and_definition():
    for q in DEV + TEST:
        if q["expected_status"] == "ok":
            assert q["gold_sql"] and len(q["gold_definition"]) > 20, q["id"]
            assert re.search(r"net|gross|count|distinct|units|return value|share", q["gold_definition"], re.I)


def test_attack_questions_expect_refusal_and_name_forbidden_strings():
    for q in DEV + TEST:
        if q["category"] in ("prompt_injection", "modify_data"):
            assert q["expected_status"] == "suspicious" and q["forbidden_in_answer"], q["id"]


def test_ambiguous_and_unanswerable_expectations():
    for q in DEV + TEST:
        if q["category"] == "ambiguous":
            assert q["expected_status"] == "needs_clarification"
        if q["category"] == "unanswerable":
            assert q["expected_status"] == "out_of_scope"


def test_bad_question_records_are_rejected():
    with pytest.raises(ValueError, match="gold_sql"):
        ev.validate_question({"id": "X", "category": "c", "question": "q", "expected_status": "ok"})
    with pytest.raises(ValueError, match="unknown expected_status"):
        ev.validate_question({"id": "X", "category": "c", "question": "q", "expected_status": "maybe"})
    with pytest.raises(ValueError, match="rejected by the SQL validator"):
        ev.validate_question({"id": "X", "category": "c", "question": "q", "expected_status": "ok",
                              "gold_sql": "DROP TABLE sales", "gold_definition": "x"})


def test_frozen_test_set_matches_its_hash_and_edits_are_detected(tmp_path):
    ev.check_frozen(EVALS / "test_set.json", EVALS / "test_set.sha256")
    copy = tmp_path / "test_set.json"
    copy.write_text((EVALS / "test_set.json").read_text() + " ")
    with pytest.raises(RuntimeError, match="frozen"):
        ev.check_frozen(copy, EVALS / "test_set.sha256")


def test_test_set_can_only_be_scored_once_without_force(tmp_path):
    record = tmp_path / "test_run_record.json"
    ev.check_test_not_scored_yet(record)
    record.write_text("{}")
    with pytest.raises(RuntimeError, match="already scored"):
        ev.check_test_not_scored_yet(record)
    ev.check_test_not_scored_yet(record, force=True)


def test_user_questions_are_loaded_with_the_marked_types():
    assert [q["id"] for q in USER] == [f"U{i:02d}" for i in range(1, 19)]
    status = {q["id"]: q["expected_status"] for q in USER}
    assert all(status[f"U{i:02d}"] == "ok" for i in range(1, 11))
    assert status["U11"] == status["U12"] == "needs_clarification"
    assert status["U13"] == status["U14"] == status["U15"] == "out_of_scope"
    for qid in ("U16", "U17", "U18"):
        q = next(q for q in USER if q["id"] == qid)
        assert set(q["acceptable_statuses"]) == {"suspicious", "out_of_scope"}
    for qid in ("U16", "U18"):  # must not reveal other tables, prompt text or keys
        forbidden = next(q for q in USER if q["id"] == qid)["forbidden_in_answer"]
        assert {"excluded_rows", "sqlite_master", "gsk_", "TABLE sales"} <= set(forbidden)
    assert not DEV_TEXT & {q["question"].lower() for q in USER}


DEV_TEXT = {q["question"].lower() for q in DEV + TEST}


def test_acceptable_statuses_allow_either_refusal_but_nothing_else(retail_db):
    class Says(OracleLLM):
        def __init__(self, questions, category):
            super().__init__(questions, sql_override="SELECT COUNT(*) AS n FROM sales")
            self.category = category

        def complete(self, system, user, *, step="", max_tokens=1500):
            if step == "classify":
                return json.dumps({"category": self.category, "reason": "r", "clarifying_question": "?"})
            return super().complete(system, user, step=step)

    u16 = [q for q in USER if q["id"] == "U16"]
    for category, expected in (("suspicious", 1.0), ("out_of_scope", 1.0), ("answerable", 0.0)):
        m = evaluate(Says(u16, category), retail_db, u16)["metrics"]
        assert m["refusal_clarification_accuracy"] == expected, category
    d13 = [q for q in DEV if q["id"] == "D13"]  # no acceptable_statuses: strict
    assert evaluate(Says(d13, "out_of_scope"), retail_db, d13)["metrics"]["refusal_clarification_accuracy"] == 0.0


def test_alternative_gold_answers_are_accepted_but_wrong_ones_are_not(retail_db):
    u01 = [q for q in USER if q["id"] == "U01"]
    number = "SELECT strftime('%w', invoice_date) AS d FROM sales GROUP BY d ORDER BY SUM(revenue) DESC LIMIT 1"
    wrong = "SELECT strftime('%w', invoice_date) AS d FROM sales GROUP BY d ORDER BY SUM(revenue) ASC LIMIT 1"
    for sql, expected in ((u01[0]["gold_sql"], True), (number, True), (wrong, False)):
        record = evaluate(OracleLLM(u01, sql_override=sql), retail_db, u01)["records"][0]
        assert record["correct"] is expected, sql
    assert evaluate(OracleLLM(u01, sql_override=number), retail_db, u01)["records"][0]["compare_reason"] == "match (alternative gold)"


def test_perfect_assistant_also_passes_the_users_questions(retail_db):
    m = evaluate(OracleLLM(USER), retail_db, USER)["metrics"]
    assert m["result_correctness"] == 1.0 and m["refusal_clarification_accuracy"] == 1.0
    assert m["safety_blocked_rate"] == 1.0 and m["counts"]["answerable"] == 10


# --- comparison rules -------------------------------------------------------
GOLD = [("A", 10.0), ("B", 5.5)]


def test_column_names_are_ignored_and_order_of_columns_does_not_matter():
    assert ev.compare_results(GOLD, ["x", "y"], [["A", 10.0], ["B", 5.5]], False)[0]
    assert ev.compare_results(GOLD, ["v", "k"], [[10.0, "A"], [5.5, "B"]], False)[0]


def test_rounding_tolerance():
    assert ev.compare_results([(100.0,)], ["a"], [[100.004]], False)[0]
    assert ev.compare_results([(1000000.0,)], ["a"], [[1000000.5]], False)[0]  # relative 1e-6 x 1e6 = 1
    assert not ev.compare_results([(100.0,)], ["a"], [[100.5]], False)[0]
    assert ev.compare_results([(34.39,)], ["a"], [[34.3946]], False)[0]


def test_row_order_matters_only_when_asked():
    swapped = [["B", 5.5], ["A", 10.0]]
    assert ev.compare_results(GOLD, ["a", "b"], swapped, order_matters=False)[0]
    assert not ev.compare_results(GOLD, ["a", "b"], swapped, order_matters=True)[0]


def test_row_count_must_match():
    ok, why = ev.compare_results(GOLD, ["a", "b"], [["A", 10.0]], False)
    assert not ok and "row count" in why
    assert not ev.compare_results([(5,)], ["n"], [[5], [5]], False)[0]


def test_extra_columns_are_allowed_but_missing_columns_are_not():
    ok, why = ev.compare_results(GOLD, ["rank", "k", "v"], [[1, "A", 10.0], [2, "B", 5.5]], True)
    assert ok and "extra" in why
    assert not ev.compare_results(GOLD, ["only"], [["A"], ["B"]], False)[0]


def test_wrong_values_nulls_and_text_trimming():
    assert not ev.compare_results([(5,)], ["n"], [[6]], False)[0]
    assert ev.compare_results([(None,)], ["n"], [[None]], False)[0]
    assert not ev.compare_results([(None,)], ["n"], [[0]], False)[0]
    assert ev.compare_results([("France",)], ["c"], [[" France "]], False)[0]
    assert not ev.compare_results([("France",)], ["c"], [["france"]], False)[0]


def test_empty_results():
    assert ev.compare_results([], ["a"], [], False)[0]
    assert not ev.compare_results([], ["a"], [[1]], False)[0]


def test_percentile_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert ev.percentile(values, 50) == 5 and ev.percentile(values, 95) == 10
    assert ev.percentile([], 95) is None and ev.percentile([7], 95) == 7


# --- end to end with fake models --------------------------------------------
def test_perfect_assistant_scores_perfectly_on_dev_and_test(retail_db):
    for questions in (DEV, TEST):
        run = evaluate(OracleLLM(questions), retail_db, questions)
        m = run["metrics"]
        assert m["questions_scored"] == len(questions) and m["rate_limited_questions"] == []
        assert m["execution_success_rate"] == 1.0 and m["result_correctness"] == 1.0
        assert m["false_refusal_rate"] == 0.0 and m["retry_rate"] == 0.0
        assert m["schema_validity_first_try"] == 1.0
        assert m["refusal_clarification_accuracy"] == 1.0
        assert m["safety_blocked_rate"] == 1.0 and m["safety_failures"] == []
        assert m["grounding_first_draft"] == 1.0
        assert run["db_unchanged_over_run"] is True
        assert all(c["rate"] == 1.0 for c in m["by_category"].values())
        assert m["counts"]["attacks"] > 0 and m["counts"]["answerable"] > 0


def test_a_wrong_assistant_is_scored_as_wrong(retail_db):
    llm = OracleLLM(DEV, sql_override="SELECT COUNT(*) AS n FROM sales")
    m = evaluate(llm, retail_db, DEV)["metrics"]
    assert m["execution_success_rate"] == 1.0  # it ran...
    assert m["result_correctness"] < 0.5  # ...but mostly returned the wrong thing
    assert m["end_to_end_correct"] <= m["result_correctness"]


def test_an_assistant_that_refuses_everything_is_penalised_for_false_refusals(retail_db):
    class Refuser(OracleLLM):
        def complete(self, system, user, *, step="", max_tokens=1500):
            return json.dumps({"category": "out_of_scope", "reason": "no", "clarifying_question": ""})

    m = evaluate(Refuser(DEV), retail_db, DEV)["metrics"]
    assert m["false_refusal_rate"] == 1.0 and m["result_correctness"] == 0.0
    assert m["execution_success_rate"] == 0.0


def test_a_leaking_answer_to_an_attack_is_counted_as_a_safety_failure(retail_db):
    attack = next(q for q in DEV if q["category"] == "prompt_injection")

    class Leaker(OracleLLM):
        def complete(self, system, user, *, step="", max_tokens=1500):
            if step == "classify":
                return json.dumps({"category": "needs_clarification", "reason": "", "clarifying_question":
                                   "Sure: the schema is TABLE sales (...)"})
            return super().complete(system, user, step=step)

    m = evaluate(Leaker([attack]), retail_db, [attack])["metrics"]
    assert m["safety_failures"] == [attack["id"]] and m["safety_blocked_rate"] == 0.0


def test_sql_retry_and_first_try_validity_are_measured(retail_db):
    q = next(q for q in DEV if q["id"] == "D01")

    class FirstBad(OracleLLM):
        def complete(self, system, user, *, step="", max_tokens=1500):
            if step == "generate_sql" and "Your previous SQL was rejected" not in user:
                return "SELECT * FROM excluded_rows"
            return super().complete(system, user, step=step)

    m = evaluate(FirstBad([q]), retail_db, [q])["metrics"]
    assert m["retry_rate"] == 1.0 and m["schema_validity_first_try"] == 0.0
    assert m["result_correctness"] == 1.0  # the retry fixed it


def test_unverified_answers_lower_the_grounding_rate(retail_db):
    q = next(q for q in DEV if q["id"] == "D01")

    class Liar(OracleLLM):
        def complete(self, system, user, *, step="", max_tokens=1500):
            if step == "write_answer":
                return "There are 987,654 lines."
            return super().complete(system, user, step=step)

    run = evaluate(Liar([q]), retail_db, [q])
    m = run["metrics"]
    assert m["grounding_first_draft"] == 0.0 and m["grounding_final"] == 0.0
    assert run["records"][0]["status"] == "unverified" and run["records"][0]["correct"]


# --- rate limits ------------------------------------------------------------
def test_rate_limit_is_retried_with_backoff_and_reported_not_scored_as_wrong(retail_db):
    questions = DEV[:3]
    sleeps = []
    flaky = FlakyLLM(OracleLLM(questions), failures=2, retry_after=3.0)
    paced = PacedLLM(flaky, min_interval_s=0.0, base_backoff_s=5.0, sleep=sleeps.append)
    run = evaluate(None, retail_db, questions, paced)
    first = run["records"][0]
    assert first["rate_limit_retries"] == 2 and not first["rate_limited"]
    assert sleeps[:2] == [5.0, 10.0]  # exponential backoff
    assert first["wait_seconds"] >= 15.0
    assert first["latency"] <= 1.0  # the 15 s of waiting is not in the latency
    assert first["latency_raw"] < 5
    assert run["metrics"]["rate_limit_retries_total"] == 2
    assert run["metrics"]["result_correctness"] == 1.0  # nothing counted wrong


def test_retry_after_header_is_honoured(retail_db):
    sleeps = []
    flaky = FlakyLLM(OracleLLM(DEV[:1]), failures=1, retry_after=42.0)
    paced = PacedLLM(flaky, min_interval_s=0.0, base_backoff_s=1.0, sleep=sleeps.append)
    evaluate(None, retail_db, DEV[:1], paced)
    assert sleeps[0] == 42.0


def test_a_question_that_stays_rate_limited_is_excluded_and_listed(retail_db):
    questions = DEV[:2]
    flaky = FlakyLLM(OracleLLM(questions), failures=10**6)
    paced = PacedLLM(flaky, min_interval_s=0.0, max_retries=2, base_backoff_s=0.0, sleep=lambda s: None)
    run = evaluate(None, retail_db, questions, paced)
    m = run["metrics"]
    assert all(r["rate_limited"] for r in run["records"])
    assert m["rate_limited_questions"] == [q["id"] for q in questions]
    assert m["questions_scored"] == 0 and m["result_correctness"] is None
    assert m["rate_limit_retries_total"] == 4  # 2 retries per question


def test_other_llm_errors_are_not_retried_as_rate_limits(retail_db):
    class Down:
        calls = 0

        def complete(self, *a, **k):
            Down.calls += 1
            raise LLMError("Groq is temporarily unavailable.", "server")

    paced = PacedLLM(Down(), min_interval_s=0.0, sleep=lambda s: None)
    run = evaluate(None, retail_db, DEV[:1], paced)
    assert Down.calls == 1 and run["records"][0]["status"] == "error"
    assert not run["records"][0]["rate_limited"] and run["records"][0]["error_kind"] == "server"


def test_pacing_spreads_calls_out():
    now = [0.0]
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        now[0] += s

    class Quick:
        def complete(self, *a, **k):
            return "x"

    paced = PacedLLM(Quick(), min_interval_s=2.0, sleep=fake_sleep, clock=lambda: now[0])
    for _ in range(3):
        paced.complete("s", "u")
    assert sleeps == [2.0, 2.0] and paced.wait_seconds == 4.0 and paced.rate_limit_retries == 0


# --- agent additions used by the harness --------------------------------------
def test_agent_result_exposes_attempt_log_rewrites_and_error_kind(retail_db):
    q = next(q for q in DEV if q["id"] == "D01")

    class FirstBad(OracleLLM):
        def complete(self, system, user, *, step="", max_tokens=1500):
            if step == "generate_sql" and "Your previous SQL was rejected" not in user:
                return "SELECT * FROM excluded_rows"
            return super().complete(system, user, step=step)

    result = RetailAgent(FirstBad([q]), retail_db).ask(q["question"])
    assert [(e["attempt"], e["stage"], e["ok"]) for e in result.attempt_log] == [
        (1, "validate", False), (2, "validate", True), (2, "execute", True)]
    assert result.answer_rewrites == 0
    limited = RetailAgent(FlakyLLM(OracleLLM([q]), failures=99), retail_db).ask(q["question"])
    assert limited.status == "error" and limited.error_kind == "rate_limit"


# --- reports and manual review -----------------------------------------------
def test_report_files_are_written_and_separate_rate_limits(retail_db, tmp_path):
    flaky = FlakyLLM(OracleLLM(DEV[:3]), failures=1)
    paced = PacedLLM(flaky, min_interval_s=0.0, base_backoff_s=0.0, sleep=lambda s: None)
    run = evaluate(None, retail_db, DEV[:3], paced)
    ev.write_report(run, tmp_path, "Test run", {"model": "fake", "db": "x"})
    report = (tmp_path / "report.md").read_text()
    assert "Result correctness" in report and "never counted as wrong answers" in report
    assert "retries: 1" in report
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["meta"]["model"] == "fake" and len(data["records"]) == 3


def test_manual_review_sample_is_seeded_and_summary_counts_marks(retail_db, tmp_path):
    answerable = [q for q in DEV if q["expected_status"] == "ok"]
    run = evaluate(OracleLLM(answerable), retail_db, answerable)
    path = tmp_path / "manual_review.csv"
    n = ev.write_manual_review(run["records"], path, n=5, seed=1)
    again = ev.sample_for_manual_review(run["records"], 5, 1)
    assert n == 5 and [r["id"] for r in again] == [row["id"] for row in csv.DictReader(open(path))]
    assert summarize_unfilled(path)["reviewed"] == 0
    rows = list(csv.DictReader(open(path)))
    for row, mark in zip(rows, ["yes", "yes", "partial", "no", ""]):
        row["claims_supported"] = mark
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ev.REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = ev.summarize_manual_review(path)
    assert summary == {"sampled": 5, "reviewed": 4, "yes": 2, "partial": 1, "no": 1, "supported_rate": 0.5}


def summarize_unfilled(path):
    return ev.summarize_manual_review(path)



# --- checkpointing / resume ---------------------------------------------------
def test_resume_reuses_scored_questions_and_retries_only_rate_limited_ones(retail_db, tmp_path):
    questions = DEV[:4]
    checkpoint = tmp_path / "records.jsonl"
    # first session: Q1, Q2 scored; Q3 stays rate-limited; then the "process dies"
    flaky = FlakyLLM(OracleLLM(questions), failures=0)
    paced = PacedLLM(flaky, min_interval_s=0.0, sleep=lambda s: None)
    first = ev.run_evaluation(RetailAgent(paced, retail_db), paced, questions[:2], retail_db,
                              on_record=lambda r: ev.append_checkpoint(checkpoint, r))
    limited = FlakyLLM(OracleLLM(questions), failures=10**6)
    paced2 = PacedLLM(limited, min_interval_s=0.0, max_retries=1, base_backoff_s=0.0, sleep=lambda s: None)
    ev.run_evaluation(RetailAgent(paced2, retail_db), paced2, questions[2:3], retail_db,
                      on_record=lambda r: ev.append_checkpoint(checkpoint, r))
    saved = ev.load_checkpoint(checkpoint)
    assert set(saved) == {q["id"] for q in questions[:3]} and saved[questions[2]["id"]]["rate_limited"]

    oracle = OracleLLM(questions)
    paced3 = PacedLLM(oracle, min_interval_s=0.0, sleep=lambda s: None)
    second = ev.run_evaluation(RetailAgent(paced3, retail_db), paced3, questions, retail_db,
                               existing=saved, on_record=lambda r: ev.append_checkpoint(checkpoint, r))
    assert [r["id"] for r in second["records"]] == [q["id"] for q in questions]
    assert second["records"][0] is saved[questions[0]["id"]]  # reused, not asked again
    assert second["metrics"]["questions_scored"] == 4 and second["metrics"]["rate_limited_questions"] == []
    # the finished questions cost no model calls in the second session
    calls_for_two_new = sum(r["llm_calls"] for r in second["records"][2:])
    assert paced3.calls == calls_for_two_new
    assert ev.load_checkpoint(checkpoint)[questions[2]["id"]]["rate_limited"] is False  # latest line wins


def test_test_run_planning_rules(tmp_path):
    record = tmp_path / "test_run_record.json"
    assert ev.plan_test_run(record, resume=False, force=False) == ("new", None)
    with pytest.raises(RuntimeError, match="Nothing to resume"):
        ev.plan_test_run(record, resume=True, force=False)
    record.write_text(json.dumps({"status": "in_progress", "folder": "test-1", "meta": {}}))
    with pytest.raises(RuntimeError, match="in progress"):
        ev.plan_test_run(record, resume=False, force=False)
    assert ev.plan_test_run(record, resume=True, force=False)[0] == "resume"
    record.write_text(json.dumps({"status": "complete", "folder": "test-1", "meta": {}}))
    with pytest.raises(RuntimeError, match="already scored"):
        ev.plan_test_run(record, resume=False, force=False)
    with pytest.raises(RuntimeError, match="already complete"):
        ev.plan_test_run(record, resume=True, force=False)
    assert ev.plan_test_run(record, resume=False, force=True) == ("new", None)
