"""One test per cleaning rule, using tiny hand-made frames (no raw data needed)."""

import sqlite3

import pandas as pd
import pytest

from src.retail_analytics.cleaning import EXCLUSION_RULES, clean_transactions
from src.retail_analytics.database import write_database
from src.retail_analytics.ingestion import load_transactions

GOOD = {
    "InvoiceNo": "536365",
    "StockCode": "85123A",
    "Description": "WHITE HANGING HEART",
    "Quantity": 6,
    "InvoiceDate": "2010-12-01 08:26:00",
    "UnitPrice": 2.55,
    "CustomerID": "17850",
    "Country": "United Kingdom",
}


def make(*overrides: dict) -> pd.DataFrame:
    rows = [{**GOOD, **o} for o in overrides]
    frame = pd.DataFrame(rows)
    for col in ("InvoiceNo", "StockCode", "Description", "CustomerID", "Country"):
        frame[col] = frame[col].astype("string")
    frame.insert(0, "source_row_number", range(2, len(frame) + 2))
    return frame


def reasons_of(result) -> str:
    assert len(result.excluded) == 1, result.excluded
    return result.excluded["exclusion_reasons"].iloc[0]


def test_valid_sale_goes_to_sales_with_revenue():
    result = clean_transactions(make({}))
    assert len(result.sales) == 1 and result.returns.empty and result.excluded.empty
    row = result.sales.iloc[0]
    assert row["revenue"] == pytest.approx(6 * 2.55)
    assert row["invoice_date"] == "2010-12-01"
    assert row["invoice_datetime"] == "2010-12-01 08:26:00"
    assert row["has_customer_id"] == 1 and row["customer_id"] == "17850"


def test_missing_customer_id_is_kept_and_flagged():
    result = clean_transactions(make({"CustomerID": pd.NA}))
    assert len(result.sales) == 1 and result.excluded.empty
    row = result.sales.iloc[0]
    assert row["has_customer_id"] == 0 and pd.isna(row["customer_id"])
    assert result.no_customer_id_counts["sales"] == 1


def test_malformed_customer_id_is_nulled_not_dropped():
    result = clean_transactions(make({"CustomerID": "ABC1"}))
    assert len(result.sales) == 1
    assert result.sales.iloc[0]["has_customer_id"] == 0
    assert result.invalid_customer_id_format == 1


def test_cancellation_invoice_goes_to_returns():
    result = clean_transactions(make({"InvoiceNo": "C536379", "Quantity": -2}))
    assert result.sales.empty and result.excluded.empty
    row = result.returns.iloc[0]
    assert row["quantity"] == -2
    assert row["return_value"] == pytest.approx(2 * 2.55)  # positive money


def test_negative_quantity_without_c_prefix_is_a_return():
    result = clean_transactions(make({"InvoiceNo": "536400", "Quantity": -1}))
    assert len(result.returns) == 1 and result.sales.empty


def test_cancellation_with_positive_quantity_is_excluded():
    result = clean_transactions(make({"InvoiceNo": "C536379", "Quantity": 3}))
    assert reasons_of(result) == "cancellation_with_positive_quantity"


def test_extra_exact_duplicate_excluded_first_copy_kept():
    result = clean_transactions(make({}, {}, {}))
    assert len(result.sales) == 1
    assert result.sales.iloc[0]["source_row_number"] == 2  # first copy kept
    assert list(result.excluded["primary_reason"]) == ["exact_duplicate_extra_copy"] * 2


def test_same_invoice_different_line_is_not_a_duplicate():
    result = clean_transactions(make({}, {"StockCode": "71053"}))
    assert len(result.sales) == 2 and result.excluded.empty


@pytest.mark.parametrize("code", ["POST", "M", "DOT", "C2", "BANK CHARGES", "post", "m"])
def test_non_product_stock_codes_excluded(code):
    result = clean_transactions(make({"StockCode": code}))
    assert reasons_of(result) == "non_product_line"


@pytest.mark.parametrize(
    "code", ["gift_0001_20", "GIFT_0001_50", "22016", "23444", "23574"]
)
def test_vouchers_and_fee_lines_are_non_product(code):
    result = clean_transactions(make({"StockCode": code}))
    assert reasons_of(result) == "non_product_line"


@pytest.mark.parametrize("code", ["DCGSSBOY", "DCGS0076", "PADS", "23443", "22015"])
def test_look_alike_codes_stay_products(code):
    result = clean_transactions(make({"StockCode": code}))
    assert len(result.sales) == 1 and result.excluded.empty


def test_huge_cancelled_orders_are_kept_in_sales_and_returns():
    result = clean_transactions(
        make(
            {"InvoiceNo": "581483", "StockCode": "23843", "Quantity": 80995, "UnitPrice": 2.08},
            {"InvoiceNo": "C581484", "StockCode": "23843", "Quantity": -80995, "UnitPrice": 2.08},
        )
    )
    assert result.sales.iloc[0]["revenue"] == pytest.approx(168469.6)
    assert result.returns.iloc[0]["return_value"] == pytest.approx(168469.6)
    assert result.excluded.empty


@pytest.mark.parametrize("price", [0, -5.0])
def test_nonpositive_price_excluded(price):
    result = clean_transactions(make({"UnitPrice": price}))
    assert reasons_of(result) == "nonpositive_unit_price"


def test_zero_quantity_excluded():
    result = clean_transactions(make({"Quantity": 0}))
    assert reasons_of(result) == "zero_quantity"


def test_missing_and_invalid_invoice_excluded():
    assert reasons_of(clean_transactions(make({"InvoiceNo": pd.NA}))) == "missing_invoice_id"
    assert reasons_of(clean_transactions(make({"InvoiceNo": "A563185"}))) == "invalid_invoice_id"


def test_unparseable_date_and_numbers_excluded():
    assert reasons_of(clean_transactions(make({"InvoiceDate": "not a date"}))) == "invalid_timestamp"
    assert reasons_of(clean_transactions(make({"Quantity": "x"}))) == "invalid_quantity_value"
    assert reasons_of(clean_transactions(make({"UnitPrice": "x"}))) == "invalid_unit_price_value"


def test_missing_stock_code_excluded():
    assert reasons_of(clean_transactions(make({"StockCode": pd.NA}))) == "missing_stock_code"


def test_multiple_reasons_are_all_recorded_in_priority_order():
    result = clean_transactions(make({"StockCode": "POST", "UnitPrice": 0}))
    assert reasons_of(result) == "non_product_line|nonpositive_unit_price"
    assert result.excluded.iloc[0]["primary_reason"] == "non_product_line"


def test_non_product_return_line_is_excluded_not_returned():
    result = clean_transactions(make({"InvoiceNo": "C536379", "StockCode": "M", "Quantity": -1}))
    assert result.returns.empty and reasons_of(result) == "non_product_line"


def test_every_row_is_accounted_for_exactly_once():
    raw = make(
        {}, {}, {"StockCode": "POST"}, {"UnitPrice": 0}, {"CustomerID": pd.NA},
        {"InvoiceNo": "C536379", "Quantity": -1}, {"Quantity": 0},
    )
    result = clean_transactions(raw)
    seen = pd.concat(
        [result.sales["source_row_number"], result.returns["source_row_number"],
         result.excluded["source_row_number"]]
    )
    assert sorted(seen) == sorted(raw["source_row_number"])
    assert sum(result.primary_counts.values()) == len(result.excluded)
    assert set(result.rule_counts) == set(EXCLUSION_RULES)


def test_database_roundtrip_types_and_totals(tmp_path):
    raw = make(
        {"Quantity": 2, "UnitPrice": 10.0},
        {"InvoiceNo": "536366", "Quantity": 1, "UnitPrice": 5.0, "CustomerID": pd.NA},
        {"InvoiceNo": "C536367", "Quantity": -1, "UnitPrice": 4.0},
        {"StockCode": "POST"},
    )
    db = tmp_path / "retail.sqlite"
    write_database(clean_transactions(raw), db)
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 2
    assert con.execute("SELECT SUM(revenue) FROM sales").fetchone()[0] == 25.0
    assert con.execute("SELECT SUM(return_value) FROM returns").fetchone()[0] == 4.0
    assert con.execute("SELECT COUNT(*) FROM excluded_rows").fetchone()[0] == 1
    assert con.execute("SELECT typeof(invoice_date), typeof(quantity), typeof(has_customer_id) FROM sales LIMIT 1").fetchone() == ("text", "integer", "integer")
    assert con.execute("SELECT COUNT(*) FROM sales WHERE has_customer_id=0 AND customer_id IS NULL").fetchone()[0] == 1


def test_loader_adds_excel_row_numbers_and_rejects_bad_input(tmp_path):
    path = tmp_path / "mini.xlsx"
    pd.DataFrame([{k: v for k, v in GOOD.items()}] * 2).to_excel(path, index=False)
    frame = load_transactions(path)
    assert list(frame["source_row_number"]) == [2, 3]
    with pytest.raises(FileNotFoundError):
        load_transactions(tmp_path / "missing.xlsx")
    bad = tmp_path / "bad.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(bad, index=False)
    with pytest.raises(ValueError):
        load_transactions(bad)
