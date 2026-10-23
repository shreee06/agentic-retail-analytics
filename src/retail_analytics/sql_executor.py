"""Run validated SQL against the retail database, read-only and bounded.

Defence in depth - each layer works even if the validator has a bug:

1. file opened with ``mode=ro`` (SQLite refuses every write)
2. ``PRAGMA query_only = ON``
3. an authorizer that only permits SELECT, reads of allowlisted tables and
   allowlisted functions (this blocks PRAGMA, ATTACH, sqlite_master, etc.)
4. a wall-clock timeout enforced through a progress handler
5. result rows capped; error messages never contain file paths
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .sql_safety import ALLOWED_TABLES, MAX_ROWS

DEFAULT_TIMEOUT_S = 5.0

# SQLite's own names for the functions we allow (the validator checks the
# parsed form; this is the second, independent check).
AUTHORIZED_FUNCTIONS = frozenset(
    {
        "count", "sum", "avg", "min", "max", "total", "group_concat", "abs",
        "round", "coalesce", "ifnull", "nullif", "iif", "length", "lower",
        "upper", "substr", "substring", "trim", "ltrim", "rtrim", "replace",
        "instr", "typeof", "date", "time", "datetime", "julianday", "strftime",
        "row_number", "rank", "dense_rank", "ntile", "lag", "lead",
        "first_value", "last_value", "percent_rank", "cume_dist", "like",
        "glob", "ceil", "ceiling", "floor",
    }
)


class QueryExecutionError(Exception):
    """Raised with a message that is safe to show to the user or the model."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool  # True when more rows existed than max_rows


FORBIDDEN_TABLE_PREFIXES = ("sqlite_", "pragma_", "dbstat", "json_", "generate_series")


def _make_authorizer(real_tables: frozenset[str]):
    """Build the authorizer. CTEs and subqueries show up as reads of tables named
    after them, so instead of listing every legal name we deny every real table
    that is not allowlisted, plus SQLite's internal and table-valued names."""
    forbidden = real_tables - ALLOWED_TABLES

    def authorizer(action, arg1, arg2, db_name, source):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:  # arg1 = table, arg2 = column
            table = (arg1 or "").lower()
            if table in forbidden or table.startswith(FORBIDDEN_TABLE_PREFIXES):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            ok = (arg2 or "").lower() in AUTHORIZED_FUNCTIONS
            return sqlite3.SQLITE_OK if ok else sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_DENY  # PRAGMA, ATTACH, INSERT, UPDATE, DDL, ...

    return authorizer


def _safe_message(error: Exception) -> str:
    """Short message with any path-like text removed."""
    if isinstance(error, sqlite3.OperationalError) and "interrupted" in str(error).lower():
        return "The query took too long and was stopped."
    text = re.sub(r"(/[\w.\-~]+)+|[A-Za-z]:\\[^\s]+", "<path>", str(error))
    text = " ".join(text.split())[:200]
    return f"The query could not be run: {text}" if text else "The query could not be run."


def execute_readonly(
    db_path: str | Path,
    sql: str,
    max_rows: int = MAX_ROWS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> QueryResult:
    """Run one SELECT and return at most ``max_rows`` rows."""
    path = Path(db_path)
    if not path.is_file():
        raise QueryExecutionError("The database is not available.")
    max_rows = max(1, min(int(max_rows), MAX_ROWS))

    connection = None
    try:
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=timeout_s
        )
        connection.execute("PRAGMA query_only = ON")
        real_tables = frozenset(
            name.lower() for (name,) in connection.execute("SELECT name FROM sqlite_master")
        )
        connection.set_authorizer(_make_authorizer(real_tables))
        deadline = time.monotonic() + timeout_s
        # Called every 10,000 SQLite VM steps; a non-zero return aborts the query.
        connection.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0, 10_000
        )
        cursor = connection.execute(sql)  # sqlite3 itself rejects >1 statement
        columns = [d[0] for d in (cursor.description or [])]
        rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.Error as error:
        raise QueryExecutionError(_safe_message(error)) from None
    except Exception:  # never leak internals
        raise QueryExecutionError("The query could not be run.") from None
    finally:
        if connection is not None:
            connection.close()
    return QueryResult(columns=columns, rows=rows[:max_rows], truncated=len(rows) > max_rows)
