"""Check every gold SQL answer against an independent pandas calculation.

The gold SQL is run on retail.sqlite; the pandas version recomputes the same
number from the cleaned tables loaded as DataFrames (no SQL aggregation). A
mismatch means the gold answer or the pandas check is wrong.

Run: env -u PYTHONPATH .venv/bin/python scripts/verify_gold.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retail_analytics.evaluation import compare_results, load_questions, run_gold  # noqa: E402

DB = PROJECT_ROOT / "data" / "processed" / "retail.sqlite"
EVALS = PROJECT_ROOT / "evals"


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(f"{DB.resolve().as_uri()}?mode=ro", uri=True)
    sales = pd.read_sql("SELECT * FROM sales", con)
    returns = pd.read_sql("SELECT * FROM returns", con)
    con.close()
    for frame in (sales, returns):
        frame["month"] = frame["invoice_date"].str[:7]
    return sales, returns


def net(s: pd.DataFrame, r: pd.DataFrame) -> float:
    return s["revenue"].sum() - r["return_value"].sum()


def net_in_month(s, r, month):
    return net(s[s.month == month], r[r.month == month])


def customer_net(s, r):
    known_s, known_r = s[s.has_customer_id == 1], r[r.has_customer_id == 1]
    spend = known_s.groupby("customer_id")["revenue"].sum()
    back = known_r.groupby("customer_id")["return_value"].sum()
    return spend.sub(back, fill_value=0).sort_values(ascending=False)


def country_net(s, r):
    return s.groupby("country")["revenue"].sum().sub(r.groupby("country")["return_value"].sum(), fill_value=0)


def month_net(s, r):
    return s.groupby("month")["revenue"].sum().sub(r.groupby("month")["return_value"].sum(), fill_value=0)


def pct(a: float, b: float) -> float:
    return round((b - a) / a * 100, 2)


def top(series: pd.Series, n: int, ascending=False):
    ordered = series.sort_values(ascending=ascending)
    assert n >= len(ordered) or ordered.iloc[n - 1] != ordered.iloc[n], "tie at the cut-off"
    return [(k, round(float(v), 2) if isinstance(v, float) else int(v)) for k, v in ordered.head(n).items()]


CHECKS = {
    "D01": lambda s, r: [(len(s),)],
    "D02": lambda s, r: [(net_in_month(s, r, "2010-12"),)],
    "D03": lambda s, r: top(s.groupby("stock_code")["quantity"].sum(), 3),
    "D04": lambda s, r: [(s[s.country == "Spain"].invoice_no.nunique(),)],
    "D05": lambda s, r: [(net(s[(s.country == "United Kingdom") & (s.invoice_date >= "2011-01-01")],
                              r[(r.country == "United Kingdom") & (r.invoice_date >= "2011-01-01")]),)],
    "D06": lambda s, r: [(s[(s.has_customer_id == 1) & (s.country == "France")].customer_id.nunique(),)],
    "D07": lambda s, r: top(customer_net(s, r), 1),
    "D08": lambda s, r: [(r[r.country == "Germany"].return_value.sum(),)],
    "D09": lambda s, r: [((r.stock_code == "22423").sum(),)],
    "D10": lambda s, r: [(pct(net_in_month(s, r, "2011-02"), net_in_month(s, r, "2011-03")),)],
    "D15": lambda s, r: [(s[s.country == "Australia"].revenue.sum(),)],
    "T01": lambda s, r: [(net(s, r),)],
    "T02": lambda s, r: [(s.invoice_no.nunique(),)],
    "T03": lambda s, r: [(s.quantity.sum(),)],
    "T04": lambda s, r: [(s.revenue.sum(),)],
    "T05": lambda s, r: [(s.stock_code.nunique(),)],
    "T06": lambda s, r: [(net_in_month(s, r, "2011-03"),)],
    "T07": lambda s, r: [(net(s[s.country == "Germany"], r[r.country == "Germany"]),)],
    "T08": lambda s, r: top(country_net(s, r).drop("United Kingdom"), 5),
    "T09": lambda s, r: [(s[s.description == "WHITE HANGING HEART T-LIGHT HOLDER"].quantity.sum(),)],
    "T10": lambda s, r: [(net(s[s.invoice_date.between("2011-12-01", "2011-12-09")],
                              r[r.invoice_date.between("2011-12-01", "2011-12-09")]),)],
    "T11": lambda s, r: [(s[s.country == "France"].invoice_no.nunique(),)],
    "T12": lambda s, r: top(s[s.month == "2011-11"].groupby("stock_code")["quantity"].sum(), 3),
    "T13": lambda s, r: [(s[(s.country == "Netherlands") & (s.invoice_date >= "2011-01-01")].revenue.sum(),)],
    "T14": lambda s, r: [(s[s.has_customer_id == 1].customer_id.nunique(),)],
    "T15": lambda s, r: [((s[s.has_customer_id == 1].groupby("customer_id").invoice_no.nunique() > 5).sum(),)],
    "T16": lambda s, r: top(customer_net(s, r), 5),
    "T17": lambda s, r: [(round(len(s[s.has_customer_id == 1][["customer_id", "invoice_no"]].drop_duplicates())
                                / s[s.has_customer_id == 1].customer_id.nunique(), 2),)],
    "T18": lambda s, r: [(s[(s.has_customer_id == 1) & (s.country == "Germany")].customer_id.nunique(),)],
    "T19": lambda s, r: [(round((s.has_customer_id == 0).mean() * 100, 2),)],
    "T20": lambda s, r: [(r.return_value.sum(),)],
    "T21": lambda s, r: [(len(r),)],
    "T22": lambda s, r: top(r.groupby("stock_code")["return_value"].sum(), 1),
    "T23": lambda s, r: top(r.groupby("country")["return_value"].sum(), 1),
    "T24": lambda s, r: [(pct(net_in_month(s, r, "2011-10"), net_in_month(s, r, "2011-11")),)],
    "T25": lambda s, r: top(month_net(s, r), 1),
    "T26": lambda s, r: top(month_net(s, r).loc["2011-01":"2011-11"], 1, ascending=True),
    "T27": lambda s, r: [(pct(s[s.month.isin(["2011-01", "2011-02", "2011-03"])].revenue.sum(),
                              s[s.month.isin(["2011-04", "2011-05", "2011-06"])].revenue.sum()),)],
}


def stock_codes(series: pd.Series, n: int):
    ordered = series.sort_values(ascending=False)
    assert ordered.iloc[n - 1] != ordered.iloc[n], "tie at the cut-off"
    return [(k,) for k in ordered.head(n).index]


def weekday_revenue(s):
    return s.groupby(pd.to_datetime(s.invoice_date).dt.dayofweek)["revenue"].sum()  # 0 = Monday


def customers_by_year(s):
    known = s[s.has_customer_id == 1]
    return known.groupby("customer_id")["invoice_date"].agg(lambda d: ((d < "2011-01-01").any(), (d >= "2011-01-01").any()))


CHECKS.update({
    "U01": lambda s, r: [(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday_revenue(s).idxmax()],)],
    "U02": lambda s, r: [(sum(a and b for a, b in customers_by_year(s)),)],
    "U03": lambda s, r: stock_codes(s.groupby("stock_code")["revenue"].sum().sub(r.groupby("stock_code")["return_value"].sum(), fill_value=0), 5),
    "U04": lambda s, r: [(round(net(s[s.country == "United Kingdom"], r[r.country == "United Kingdom"]) / net(s, r) * 100, 2),)],
    "U05": lambda s, r: [((s.groupby("invoice_no")["stock_code"].nunique() > 10).sum(),)],
    "U06": lambda s, r: [(_aov_country(s),)],
    "U07": lambda s, r: [((s[s.has_customer_id == 1].groupby("customer_id")["invoice_no"].nunique() == 1).sum(),)],
    "U08": lambda s, r: [(int(s.assign(h=s.invoice_datetime.str[11:13].astype(int)).groupby("h")["invoice_no"].nunique().idxmax()),)],
    "U09": lambda s, r: stock_codes(r[r.invoice_date >= "2011-01-01"].groupby("stock_code")["return_value"].sum(), 3),
    "U10": lambda s, r: [(r.stock_code.nunique(),)],
})
ALT_CHECKS = {  # independent pandas versions of the accepted alternative answers
    "U01": lambda s, r: [(str((weekday_revenue(s).idxmax() + 1) % 7),)],  # strftime %w: 0 = Sunday
    "U04": lambda s, r: [(round(net(s[s.country == "United Kingdom"], r[r.country == "United Kingdom"]) / net(s, r), 4),)],
}


def _aov_country(s):
    g = s.groupby("country").agg(revenue=("revenue", "sum"), orders=("invoice_no", "nunique"))
    g = g[g.orders >= 50]
    aov = (g.revenue / g.orders).sort_values(ascending=False)
    assert aov.iloc[0] != aov.iloc[1], "tie"
    return aov.index[0]


def main() -> int:
    sales, returns = load_frames()
    rows, failures = [], 0
    for name in ("dev_set.json", "test_set.json", "test_set_user.json"):
        for q in load_questions(EVALS / name):
            if q["expected_status"] != "ok":
                continue
            gold = run_gold(DB, q["gold_sql"])
            pandas_value = [tuple(row) for row in CHECKS[q["id"]](sales, returns)]
            ok, why = compare_results(gold, ["c"] * len(pandas_value[0]), [list(r) for r in pandas_value], q["order_matters"])
            failures += not ok
            status = "OK" if ok else f"MISMATCH ({why})"
            for alt_sql in q.get("gold_alt_sql", []):  # alternatives are verified too
                alt_gold = run_gold(DB, alt_sql)
                alt_pandas = [tuple(row) for row in ALT_CHECKS[q["id"]](sales, returns)]
                alt_ok, alt_why = compare_results(alt_gold, ["c"] * len(alt_pandas[0]), [list(r) for r in alt_pandas], False)
                failures += not alt_ok
                status += f"; alt {alt_gold[0][0]} vs pandas {alt_pandas[0][0]}: " + ("OK" if alt_ok else f"MISMATCH ({alt_why})")
            rows.append((q["id"], q["question"], q["gold_definition"], gold, pandas_value, status))

    lines = ["# Gold answers: SQL vs independent pandas check", "",
             "Every value below was produced twice: by the gold SQL on retail.sqlite and by a separate pandas",
             "calculation on the cleaned tables (scripts/verify_gold.py). Regenerate with that script.", "",
             "| ID | Question | Gold definition | Gold value (SQL) | pandas value | Check |", "|---|---|---|---|---|---|"]
    for qid, question, definition, gold, pv, status in rows:
        fmt = lambda v: "; ".join(", ".join(str(c) for c in row) for row in v)
        lines.append(f"| {qid} | {question} | {definition} | {fmt(gold)} | {fmt(pv)} | {status} |")
    (EVALS / "gold_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n{len(rows)} gold answers checked, {failures} mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
