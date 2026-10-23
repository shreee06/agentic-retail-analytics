"""Safety tests. The attack and legitimate sets live in evals/ so the final
evaluation can reuse exactly the same files."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from src.retail_analytics.sql_executor import QueryExecutionError, execute_readonly
from src.retail_analytics.sql_safety import (
    ALLOWED_COLUMNS,
    MAX_ROWS,
    validate_sql,
)

EVALS = Path(__file__).resolve().parents[1] / "evals"
ATTACKS = json.loads((EVALS / "sql_attacks.json").read_text())
LEGIT = json.loads((EVALS / "legit_queries.json").read_text())
REAL_DB = Path(__file__).resolve().parents[1] / "data" / "processed" / "retail.sqlite"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_attack_set_is_big_enough_and_covers_every_category():
    assert len(ATTACKS) >= 40
    categories = {a["category"] for a in ATTACKS}
    assert {"write", "multi_statement", "forbidden_table", "schema", "pragma",
            "attach", "extension", "comment", "case", "cte_write", "malformed",
            "non_select", "resource_abuse"} <= categories
    assert len({a["id"] for a in ATTACKS}) == len(ATTACKS)


@pytest.mark.parametrize("attack", ATTACKS, ids=[a["id"] for a in ATTACKS])
def test_validator_blocks_attack(attack):
    result = validate_sql(attack["sql"])
    assert not result.ok, f"{attack['id']} got through: {attack['sql']!r}"
    assert result.sql is None and result.error


@pytest.mark.parametrize("query", LEGIT, ids=[q["id"] for q in LEGIT])
def test_validator_accepts_legitimate_query(query):
    result = validate_sql(query["sql"])
    assert result.ok, f"{query['id']} rejected: {result.error}"
    assert result.sql


@pytest.mark.parametrize("query", LEGIT, ids=[q["id"] for q in LEGIT])
def test_legitimate_query_runs_through_both_layers(query, retail_db):
    checked = validate_sql(query["sql"])
    result = execute_readonly(retail_db, checked.sql)
    assert len(result.rows) <= MAX_ROWS


@pytest.mark.skipif(not REAL_DB.is_file(), reason="real database not built")
@pytest.mark.parametrize("query", LEGIT, ids=[q["id"] for q in LEGIT])
def test_legitimate_query_runs_on_real_database(query):
    result = execute_readonly(REAL_DB, validate_sql(query["sql"]).sql, timeout_s=30)
    assert result.columns


@pytest.mark.parametrize("attack", ATTACKS, ids=[a["id"] for a in ATTACKS])
def test_executor_alone_never_changes_the_database(attack, retail_db):
    """Even if the validator were bypassed, the executor must not write."""
    before = _digest(retail_db)
    try:
        execute_readonly(retail_db, attack["sql"], timeout_s=1.0)
    except QueryExecutionError:
        pass
    assert _digest(retail_db) == before


EXECUTOR_MUST_REFUSE = {"write", "multi_statement", "pragma", "attach", "extension", "cte_write"}


@pytest.mark.parametrize(
    "attack",
    [a for a in ATTACKS if a["category"] in EXECUTOR_MUST_REFUSE and a["sql"].strip()],
    ids=lambda a: a["id"],
)
def test_executor_alone_refuses_dangerous_statements(attack, retail_db):
    with pytest.raises(QueryExecutionError):
        execute_readonly(retail_db, attack["sql"], timeout_s=1.0)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM excluded_rows",
        "SELECT * FROM sqlite_master",
        "SELECT invoice_no FROM sales UNION SELECT sql FROM sqlite_master",
        "SELECT * FROM pragma_table_info('sales')",
        "SELECT load_extension('x')",
        "SELECT sqlite_version()",
    ],
)
def test_executor_authorizer_blocks_forbidden_reads(sql, retail_db):
    with pytest.raises(QueryExecutionError):
        execute_readonly(retail_db, sql)


def test_runaway_recursion_is_stopped_by_timeout_even_without_validator(retail_db):
    sql = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r"
    with pytest.raises(QueryExecutionError, match="too long"):
        execute_readonly(retail_db, sql, timeout_s=0.5)


def test_executor_rejects_two_statements(retail_db):
    with pytest.raises(QueryExecutionError):
        execute_readonly(retail_db, "SELECT 1; SELECT 2")


def test_connection_is_really_read_only(retail_db):
    """Open the way the executor does and show SQLite itself refuses a write."""
    con = sqlite3.connect(f"{retail_db.resolve().as_uri()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM sales")
    con.close()


def test_row_cap_and_truncated_flag(retail_db):
    checked = validate_sql("SELECT * FROM sales")
    assert checked.sql.endswith(f"LIMIT {MAX_ROWS + 1}")
    result = execute_readonly(retail_db, checked.sql, max_rows=5)
    assert len(result.rows) == 5 and result.truncated


def test_default_pipeline_reports_truncation(retail_db):
    result = execute_readonly(retail_db, validate_sql("SELECT * FROM sales").sql)
    assert len(result.rows) == MAX_ROWS and result.truncated  # 3,004 rows exist


def test_small_result_is_not_truncated(retail_db):
    result = execute_readonly(retail_db, validate_sql("SELECT COUNT(*) FROM returns").sql)
    assert result.rows == [(1,)] and not result.truncated


def test_literal_limit_is_kept_and_oversized_limit_is_capped():
    assert validate_sql("SELECT * FROM sales LIMIT 5").sql == "SELECT * FROM sales LIMIT 5"
    assert validate_sql("SELECT * FROM sales LIMIT 5000").sql.endswith(f"LIMIT {MAX_ROWS + 1}")
    assert validate_sql("SELECT * FROM sales", max_rows=10).sql.endswith("LIMIT 11")


def test_wrapping_keeps_order(retail_db):
    sql = validate_sql("SELECT invoice_no FROM sales WHERE has_customer_id = 1 ORDER BY revenue DESC").sql
    rows = execute_readonly(retail_db, sql).rows
    assert [r[0] for r in rows] == ["536365", "536367", "536366"]


def test_slow_query_times_out(retail_db):
    sql = validate_sql("SELECT COUNT(*) FROM sales a, sales b, sales c").sql
    with pytest.raises(QueryExecutionError, match="too long"):
        execute_readonly(retail_db, sql, timeout_s=0.5)


def test_error_messages_do_not_leak_paths(retail_db):
    missing = retail_db.parent / "nope" / "missing.sqlite"
    with pytest.raises(QueryExecutionError) as info:
        execute_readonly(missing, "SELECT 1")
    assert str(retail_db.parent) not in str(info.value)

    with pytest.raises(QueryExecutionError) as info:
        execute_readonly(retail_db, "SELECT * FROM '/etc/passwd'")
    assert "/etc" not in str(info.value) and str(retail_db.parent) not in str(info.value)


def test_validator_errors_are_short_and_internal_free():
    for attack in ATTACKS:
        error = validate_sql(attack["sql"]).error
        assert len(error) < 120
        assert "Traceback" not in error and "/home" not in error


def test_non_string_input_is_rejected():
    assert not validate_sql(None).ok
    assert not validate_sql(123).ok


def test_allowlist_matches_the_real_schema(retail_db):
    """If a column is added to the database, the allowlist must be updated."""
    con = sqlite3.connect(retail_db)
    for table, allowed in ALLOWED_COLUMNS.items():
        actual = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
        assert actual == set(allowed), table
    con.close()
