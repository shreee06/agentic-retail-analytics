"""Run the evaluation against the live Groq model.

  dev set  (for fixing prompts):  scripts/run_eval.py --set dev
  frozen test set (score ONCE):   scripts/run_eval.py --set test

The test run refuses to start if evals/test_set.json changed after freezing, or
if it was already scored (pass --force only if you will report the second run).
Rate limits are paced and retried; they are reported separately, never as errors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retail_analytics import evaluation as ev  # noqa: E402
from src.retail_analytics.agent import RetailAgent  # noqa: E402
from src.retail_analytics.llm import GroqLLM, LLMError  # noqa: E402
from src.retail_analytics.pacing import PacedLLM  # noqa: E402

DB = PROJECT_ROOT / "data" / "processed" / "retail.sqlite"
EVALS = PROJECT_ROOT / "evals"
RESULTS = EVALS / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=["dev", "test"], required=True)
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--pace", type=float, default=2.0, help="minimum seconds between model calls")
    parser.add_argument("--review-sample", type=int, default=10, help="answers to export for manual review")
    parser.add_argument("--force", action="store_true", help="allow scoring the frozen test set again")
    args = parser.parse_args()

    if args.set == "test":
        ev.check_frozen(EVALS / "test_set.json", EVALS / "test_set.sha256")
        if not args.limit:
            ev.check_test_not_scored_yet(RESULTS / "test_run_record.json", args.force)
        questions = ev.load_questions(EVALS / "test_set.json") + ev.load_questions(EVALS / "test_set_user.json")
    else:
        questions = ev.load_questions(EVALS / "dev_set.json")
    if args.limit:
        questions = questions[: args.limit]

    try:
        inner = GroqLLM.from_env(max_retries=0)  # the harness does the retrying, so it can time it
    except LLMError as error:
        sys.exit(str(error))
    paced = PacedLLM(inner, min_interval_s=args.pace)
    agent = RetailAgent(paced, DB)

    def progress(i, n, r):
        flag = "RATE-LIMITED" if r["rate_limited"] else ("pass" if r.get("correct", r.get("status_matches")) else "FAIL")
        print(f"[{i}/{n}] {r['id']:<4} {r['status']:<20} {flag:<12} {r['latency']:.1f}s", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS / f"{args.set}-{stamp}"
    started = time.perf_counter()
    run = ev.run_evaluation(agent, paced, questions, DB, progress)
    meta = {"set": args.set, "model": inner.model, "db": DB.name, "started_utc": stamp,
            "pace_s": args.pace, "wall_seconds": round(time.perf_counter() - started, 1),
            "limit": args.limit}
    ev.write_report(run, out_dir, f"Evaluation: {args.set} set", meta)
    n = ev.write_manual_review(run["records"], out_dir / "manual_review.csv", args.review_sample)
    if args.set == "test" and not args.limit:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "test_run_record.json").write_text(json.dumps({"scored_at_utc": stamp, "folder": out_dir.name}))
    print(f"\nReport: {out_dir / 'report.md'}\nManual-review sample ({n} answers): {out_dir / 'manual_review.csv'}")
    print(json.dumps(run["metrics"], indent=1))


if __name__ == "__main__":
    main()
