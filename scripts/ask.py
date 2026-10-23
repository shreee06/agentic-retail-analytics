"""Ask one question from the command line.

Run: env -u PYTHONPATH .venv/bin/python scripts/ask.py "What are the top 10 products by revenue?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retail_analytics.agent import RetailAgent  # noqa: E402
from src.retail_analytics.llm import GroqLLM, LLMError  # noqa: E402

DEFAULT_DB = PROJECT_ROOT / "data" / "processed" / "retail.sqlite"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()

    try:
        llm = GroqLLM.from_env()
    except LLMError as error:
        sys.exit(str(error))
    result = RetailAgent(llm, args.db).ask(args.question)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return
    print(f"QUESTION: {args.question}")
    print(f"MODEL:    {llm.model}")
    print(f"STATUS:   {result.status} (category={result.category}, attempts={result.attempts})")
    print(f"\nSQL USED:\n{result.sql}")
    print(f"\nROWS ({len(result.rows)}{', truncated' if result.truncated else ''}):")
    if result.columns:
        print("  " + " | ".join(result.columns))
    for row in result.rows[:15]:
        print("  " + " | ".join(str(c) for c in row))
    if len(result.rows) > 15:
        print(f"  ... {len(result.rows) - 15} more rows")
    print(f"\nANSWER:\n{result.answer}")
    if result.warnings:
        print(f"\nWARNINGS: {result.warnings}")
    print("\nTIMING:")
    for step in result.steps:
        print(f"  {step['step']:<14s} {step['seconds']:.3f}s")
    print(f"  {'total':<14s} {result.total_seconds:.3f}s")


if __name__ == "__main__":
    main()
