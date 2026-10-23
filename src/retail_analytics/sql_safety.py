"""Validate model-written SQL before it gets anywhere near the database.

The check works on the *parsed* query (sqlglot, SQLite dialect), not on keyword
matching alone. A query is accepted only if all of these hold:

* exactly one statement, and it is a SELECT (CTEs / UNION allowed)
* no comments, no bind parameters, no WITH RECURSIVE
* every table is on the table allowlist (or a CTE defined in the same query)
* every column is on the column allowlist (or an alias defined in the query)
* every function is on the function allowlist
* no write / DDL / PRAGMA / ATTACH node anywhere in the tree

The row limit is enforced by wrapping the query when it has no safe literal
LIMIT. The returned SQL is the *original text* (never sqlglot's re-rendering),
so what runs is exactly what was checked; the executor adds a second,
independent set of defences.
"""

from __future__ import annotations

from dataclasses import dataclass

import logging

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

# sqlglot logs a warning for statements it can only parse as a generic Command
# (e.g. EXPLAIN); those are rejected anyway, so keep the log quiet.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

MAX_ROWS = 1000
MAX_SQL_LENGTH = 4000

# Table and column allowlist: what the assistant may see. `excluded_rows` is an
# audit table and `sqlite_master` exposes the schema, so neither is listed.
ALLOWED_COLUMNS: dict[str, frozenset[str]] = {
    "sales": frozenset(
        {
            "source_row_number", "invoice_no", "stock_code", "description",
            "quantity", "unit_price", "revenue", "invoice_datetime",
            "invoice_date", "customer_id", "has_customer_id", "country",
        }
    ),
    "returns": frozenset(
        {
            "source_row_number", "invoice_no", "stock_code", "description",
            "quantity", "unit_price", "return_value", "invoice_datetime",
            "invoice_date", "customer_id", "has_customer_id", "country",
        }
    ),
}
ALLOWED_TABLES = frozenset(ALLOWED_COLUMNS)
ALL_ALLOWED_COLUMNS = frozenset().union(*ALLOWED_COLUMNS.values())
# Names a CTE may not take: it could shadow or confuse the table allowlist.
RESERVED_NAMES = ALLOWED_TABLES | {"excluded_rows"}

# sqlglot node keys for the SQL functions we allow (what sqlglot calls them),
# e.g. SUBSTR -> "substring", IFNULL -> "coalesce", STRFTIME -> "timetostr".
ALLOWED_FUNCTION_KEYS = frozenset(
    {
        "count", "sum", "avg", "min", "max", "groupconcat", "abs", "round",
        "coalesce", "nullif", "if", "case", "length", "lower", "upper",
        "substring", "trim", "replace", "strposition", "typeof", "date",
        "timetostr", "tsordstotimestamp", "cast", "ceil", "floor",
        "rownumber", "rank", "denserank", "ntile", "lag", "lead",
        "firstvalue", "lastvalue", "percentrank", "cumedist",
    }
)
# SQLite functions sqlglot does not model; they arrive as "Anonymous" calls.
ALLOWED_ANONYMOUS_FUNCTIONS = frozenset({"total", "time", "datetime", "julianday"})

# Defence in depth: the root must be one of these, and none of these node types
# may appear anywhere in the tree (e.g. a DELETE hidden inside a CTE).
ALLOWED_ROOTS = ("Select", "Union", "Intersect", "Except", "Subquery")
FORBIDDEN_NODES = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter",
        "AlterTable", "TruncateTable", "Command", "Pragma", "Attach", "Detach",
        "Transaction", "Commit", "Rollback", "Set", "Use", "Copy", "Into",
        "Parameter", "Placeholder",
    )
    if hasattr(exp, name)
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    sql: str | None = None  # SQL to execute (original text, row limit enforced)
    error: str | None = None  # safe, short reason (no internals)


def _reject(reason: str) -> ValidationResult:
    return ValidationResult(ok=False, error=reason)


def _has_comment_or_bad_chars(sql: str) -> str | None:
    """Scan outside quoted text for comment openers and control characters."""
    if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in sql):
        return "Control characters are not allowed."
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in "'\"`":
            j = i + 1
            while j < n:
                if sql[j] == ch:
                    if j + 1 < n and sql[j + 1] == ch:  # doubled quote = escape
                        j += 2
                        continue
                    break
                j += 1
            else:
                return "Unterminated quoted text."
            i = j + 1
            continue
        if ch == "[":
            j = sql.find("]", i + 1)
            if j == -1:
                return "Unterminated bracketed name."
            i = j + 1
            continue
        if sql.startswith("--", i) or sql.startswith("/*", i):
            return "SQL comments are not allowed."
        i += 1
    return None


def _root_limit_is_safe(tree: exp.Expression) -> bool:
    """True when the root query has a plain positive integer LIMIT <= MAX_ROWS."""
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return False
    limit = tree.args.get("limit")
    if limit is None:
        return False
    value = limit.args.get("expression")
    if not (isinstance(value, exp.Literal) and not value.is_string):
        return False
    try:
        return 0 < int(value.name) <= MAX_ROWS
    except ValueError:
        return False


def validate_sql(sql: str, max_rows: int = MAX_ROWS) -> ValidationResult:
    """Return ``ok=True`` with the SQL to run, or ``ok=False`` with a safe reason."""
    if not isinstance(sql, str) or not sql.strip():
        return _reject("The query is empty.")
    if len(sql) > MAX_SQL_LENGTH:
        return _reject("The query is too long.")
    problem = _has_comment_or_bad_chars(sql)
    if problem:
        return _reject(problem)

    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except (SqlglotError, RecursionError):
        return _reject("The query could not be parsed as a single SQLite SELECT.")
    if len(statements) != 1:
        return _reject("Exactly one statement is allowed.")
    tree = statements[0]
    if type(tree).__name__ not in ALLOWED_ROOTS:
        return _reject("Only SELECT queries are allowed.")

    nodes = list(tree.walk())
    for node in nodes:
        if isinstance(node, FORBIDDEN_NODES):
            return _reject("Only read-only SELECT queries are allowed.")
        if isinstance(node, exp.With) and node.args.get("recursive"):
            return _reject("WITH RECURSIVE is not allowed.")
        if isinstance(node, exp.Select) and not node.expressions:
            return _reject("The query is incomplete.")

    # Functions: SQL names that load code, read files or leak internals
    # (load_extension, readfile, pragma_*, sqlite_version, ...) never match.
    for node in nodes:
        if isinstance(node, exp.Anonymous):
            if node.name.lower() not in ALLOWED_ANONYMOUS_FUNCTIONS:
                return _reject(f"Function '{node.name[:30]}' is not allowed.")
        elif isinstance(node, exp.Func) and node.key not in ALLOWED_FUNCTION_KEYS:
            return _reject(f"Function '{node.key[:30]}' is not allowed.")

    # Tables. CTE names are allowed as table references, but may not reuse a
    # real table name (so a table name always means the real table).
    cte_names: set[str] = set()
    for cte in tree.find_all(exp.CTE):
        name = cte.alias.lower()
        if name in RESERVED_NAMES or name.startswith("sqlite_"):
            return _reject("That CTE name is not allowed.")
        cte_names.add(name)

    table_of_alias: dict[str, set[str]] = {}  # alias/name -> real tables
    for table in tree.find_all(exp.Table):
        if table.args.get("db") or table.args.get("catalog"):
            return _reject("Schema-qualified tables are not allowed.")
        if not isinstance(table.this, exp.Identifier):
            return _reject("Table functions are not allowed.")
        name = table.name.lower()
        if name in cte_names:
            continue
        if name not in ALLOWED_TABLES:
            return _reject(f"Table '{table.name[:40]}' is not allowed.")
        table_of_alias.setdefault((table.alias or table.name).lower(), set()).add(name)

    # Columns. Names defined inside the query (aliases, CTE columns) are fine.
    defined: set[str] = set(cte_names)
    for node in nodes:
        if isinstance(node, (exp.Alias, exp.TableAlias)) and node.alias_or_name:
            defined.add(node.alias_or_name.lower())
        if isinstance(node, exp.TableAlias):
            defined.update(c.name.lower() for c in node.columns)
    for column in tree.find_all(exp.Column):
        name = column.name.lower()
        qualifier = column.table.lower() if column.table else None
        if qualifier and qualifier in table_of_alias:
            allowed_here = frozenset().union(
                *(ALLOWED_COLUMNS[t] for t in table_of_alias[qualifier])
            )
            if name not in allowed_here and name not in defined:
                return _reject(f"Column '{column.name[:40]}' is not allowed.")
        elif qualifier and qualifier not in defined:
            return _reject(f"Unknown table alias '{column.table[:40]}'.")
        elif name not in ALL_ALLOWED_COLUMNS and name not in defined:
            return _reject(f"Column '{column.name[:40]}' is not allowed.")

    # Row limit: keep the original text if it already has a safe literal LIMIT,
    # otherwise wrap it. The wrapper asks for one extra row so the executor can
    # tell the user the result was cut off (truncated=True).
    text = sql.strip().rstrip(";").rstrip()
    max_rows = min(int(max_rows), MAX_ROWS)
    if _root_limit_is_safe(tree) and int(tree.args["limit"].expression.name) <= max_rows:
        return ValidationResult(ok=True, sql=text)
    return ValidationResult(ok=True, sql=f"SELECT * FROM ({text}) LIMIT {max_rows + 1}")
