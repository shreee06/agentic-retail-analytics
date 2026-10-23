"""Run the evaluation against the live Groq model.

  dev set  (for fixing prompts):  scripts/run_eval.py --set dev
  frozen test set (score ONCE):   scripts/run_eval.py --set test
  continue an interrupted test run: scripts/run_eval.py --set test --resume

The test run refuses to start if evals/test_set.json changed after freezing, and
it can only be started once. Every finished question is written to records.jsonl
immediately, so an interruption loses nothing: --resume skips questions that
already have a score and only asks the rest. Questions that stayed rate-limited
were never scored, so --resume tries them again. Rate limits are paced and retried
and are reported separately, never as wrong answers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
RECORD = RESULTS / "test_run_record.json"


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance(model: str, pace: float) -> dict:
    return {
        "model": model,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_files": [line for line in git("status", "--porcelain").splitlines() if "evals/results" not in line],
        "test_set_sha256": ev.sha256_of(EVALS / "test_set.json"),
        "test_set_user_sha256": ev.sha256_of(EVALS / "test_set_user.json"),
        "database": DB.name,
        "database_sha256": ev.sha256_of(DB),
        "pace_s": pace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=["dev", "test"], required=True)
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test; never counts as the test run)")
    parser.add_argument("--pace", type=float, default=3.0, help="minimum seconds between model calls")
    parser.add_argument("--review-sample", type=int, default=12, help="answers to export for manual review")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted test run")
    parser.add_argument("--force", action="store_true", help="allow scoring the frozen test set again")
    args = parser.parse_args()

    official = args.set == "test" and not args.limit
    if args.set == "test":
        ev.check_frozen(EVALS / "test_set.json", EVALS / "test_set.sha256")
        questions = ev.load_questions(EVALS / "test_set.json") + ev.load_questions(EVALS / "test_set_user.json")
    else:
        questions = ev.load_questions(EVALS / "dev_set.json")
    if args.limit:
        questions = questions[: args.limit]

    try:
        inner = GroqLLM.from_env(max_retries=0)  # the harness retries, so it can time the waits
    except LLMError as error:
        sys.exit(str(error))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    existing: dict[str, dict] = {}
    meta = provenance(inner.model, args.pace)
    if official:
        action, record = ev.plan_test_run(RECORD, args.resume, args.force)
        if action == "resume":
            if record["meta"]["test_set_sha256"] != meta["test_set_sha256"] or \
               record["meta"]["test_set_user_sha256"] != meta["test_set_user_sha256"]:
                sys.exit("The question files changed since this run started; it cannot be resumed.")
            out_dir = RESULTS / record["folder"]
            existing = ev.load_checkpoint(out_dir / "records.jsonl")
            meta = {**record["meta"], "resumed_at_utc": [*record["meta"].get("resumed_at_utc", []), meta["date_utc"]]}
            print(f"Resuming {record['folder']}: {sum(not r['rate_limited'] for r in existing.values())} "
                  f"questions already scored, {len(questions) - sum(not r['rate_limited'] for r in existing.values())} to go.")
        else:
            out_dir = RESULTS / f"test-{stamp}"
            out_dir.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps({"status": "in_progress", "folder": out_dir.name, "meta": meta}, indent=1))
    else:
        out_dir = RESULTS / f"{args.set}{'-smoke' if args.set == 'test' else ''}-{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
    meta.update(set=args.set, question_count=len(questions))

    paced = PacedLLM(inner, min_interval_s=args.pace)
    agent = RetailAgent(paced, DB)

    def progress(i, n, r):
        flag = "RATE-LIMITED" if r["rate_limited"] else ("pass" if r.get("correct", r.get("status_matches")) else "FAIL")
        print(f"[{i}/{n}] {r['id']:<4} {r['status']:<20} {flag:<12} {r['latency']:.1f}s", flush=True)

    started = time.perf_counter()
    run = ev.run_evaluation(
        agent, paced, questions, DB, progress, existing=existing,
        on_record=lambda r: ev.append_checkpoint(out_dir / "records.jsonl", r),
    )
    meta["wall_seconds_this_session"] = round(time.perf_counter() - started, 1)
    ev.write_report(run, out_dir, f"Evaluation: {args.set} set", meta)
    n = ev.write_manual_review(run["records"], out_dir / "manual_review.csv", args.review_sample)
    if official:
        pending = run["metrics"]["rate_limited_questions"]
        status = "in_progress" if pending else "complete"
        RECORD.write_text(json.dumps({"status": status, "folder": out_dir.name, "meta": meta}, indent=1))
        if pending:
            print(f"\nNOT COMPLETE: {pending} stayed rate-limited and are unscored. Run again with --resume.")
    print(f"\nReport: {out_dir / 'report.md'}\nManual-review sample ({n} answers): {out_dir / 'manual_review.csv'}")
    print(json.dumps(run["metrics"], indent=1))


if __name__ == "__main__":
    main()
