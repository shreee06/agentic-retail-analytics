"""Build data/processed/retail.sqlite from the raw UCI Online Retail workbook.

Run: env -u PYTHONPATH .venv/bin/python scripts/build_db.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retail_analytics.cleaning import clean_transactions  # noqa: E402
from src.retail_analytics.database import write_database  # noqa: E402
from src.retail_analytics.ingestion import load_transactions  # noqa: E402

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
DEFAULT_DB = PROJECT_ROOT / "data" / "processed" / "retail.sqlite"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def db_totals(db_path: Path) -> dict:
    """Read the totals back from the finished database (not from memory)."""
    with sqlite3.connect(db_path) as con:
        one = lambda sql: con.execute(sql).fetchone()[0]
        return {
            "sales_rows": one("SELECT COUNT(*) FROM sales"),
            "returns_rows": one("SELECT COUNT(*) FROM returns"),
            "excluded_rows": one("SELECT COUNT(*) FROM excluded_rows"),
            "gross_revenue": one("SELECT ROUND(SUM(revenue), 2) FROM sales"),
            "gross_revenue_with_customer_id": one(
                "SELECT ROUND(SUM(revenue), 2) FROM sales WHERE has_customer_id = 1"
            ),
            "gross_revenue_without_customer_id": one(
                "SELECT ROUND(SUM(revenue), 2) FROM sales WHERE has_customer_id = 0"
            ),
            "return_value": one("SELECT ROUND(SUM(return_value), 2) FROM returns"),
            "net_revenue": one(
                "SELECT ROUND((SELECT SUM(revenue) FROM sales)"
                " - (SELECT SUM(return_value) FROM returns), 2)"
            ),
            "first_date": one("SELECT MIN(invoice_date) FROM sales"),
            "last_date": one("SELECT MAX(invoice_date) FROM sales"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    if not args.input.is_file():
        sys.exit(
            f"Raw file not found: {args.input}\n"
            "Download the UCI Online Retail dataset (see docs/data_cleaning.md) "
            "and place 'Online Retail.xlsx' in data/raw/."
        )

    raw = load_transactions(args.input)
    result = clean_transactions(raw)
    total = len(result.sales) + len(result.returns) + len(result.excluded)
    if total != len(raw):
        sys.exit(f"Row accounting failed: {total} classified vs {len(raw)} source rows")
    write_database(result, args.output)

    report = {
        "source_file": args.input.name,
        "source_sha256": sha256_file(args.input),
        "source_rows": len(raw),
        "rule_counts_overlapping": result.rule_counts,
        "primary_reason_counts": result.primary_counts,
        "no_customer_id_rows": result.no_customer_id_counts,
        "invalid_customer_id_format": result.invalid_customer_id_format,
        "database_totals": db_totals(args.output),
    }
    report_path = args.output.with_name("build_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nDatabase written to {args.output}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
