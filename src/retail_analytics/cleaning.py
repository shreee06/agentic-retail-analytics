"""Row-level cleaning rules: every source row is classified, none is dropped.

Adapted from Project 1 (my-new-project/src/data/validation.py). Each source row
ends up in exactly one of three groups:

* ``sales``     - valid product sales lines
* ``returns``   - valid cancellation / return lines
* ``excluded``  - rows that break at least one rule, with the reason(s) recorded

So ``len(sales) + len(returns) + len(excluded) == len(source)`` always holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .ingestion import EXPECTED_COLUMNS

INVOICE_PATTERN = r"(?i)C?\d{6}"
CUSTOMER_PATTERN = r"\d{5}"

# Service, fee and adjustment lines (postage, manual entries, carriage, bank
# charges, discounts, samples, ...). They are not products a customer bought.
NON_PRODUCT_STOCK_CODES = frozenset(
    {"POST", "DOT", "M", "C2", "D", "S", "B", "BANK CHARGES", "AMAZONFEE", "CRUK"}
    # Added in this project: shipping/packing fees and a gift voucher that have
    # numeric stock codes (Project 1 missed them).
    | {"23444", "23574", "22016"}
)
# Dotcomgiftshop gift vouchers (gift_0001_10 ... gift_0001_50): prepaid credit.
NON_PRODUCT_PREFIXES = ("GIFT_",)

# Exclusion rules in priority order. A row can break several; the first one
# that applies is its ``primary_reason`` so primary counts add up to the total.
EXCLUSION_RULES = (
    "missing_invoice_id",
    "invalid_invoice_id",
    "invalid_timestamp",
    "invalid_quantity_value",
    "invalid_unit_price_value",
    "missing_stock_code",
    "exact_duplicate_extra_copy",
    "non_product_line",
    "nonpositive_unit_price",
    "zero_quantity",
    "cancellation_with_positive_quantity",
)


@dataclass
class CleanResult:
    sales: pd.DataFrame
    returns: pd.DataFrame
    excluded: pd.DataFrame
    rule_counts: dict[str, int]  # rows breaking each rule (rules overlap)
    primary_counts: dict[str, int]  # rows by first rule broken (sums to excluded)
    no_customer_id_counts: dict[str, int]  # rows without a usable CustomerID
    invalid_customer_id_format: int  # non-empty IDs that are not 5 digits


def _clean_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text.eq(""))


def clean_transactions(raw: pd.DataFrame) -> CleanResult:
    invoice = _clean_text(raw["InvoiceNo"])
    stock_code = _clean_text(raw["StockCode"])
    description = _clean_text(raw["Description"])
    customer = _clean_text(raw["CustomerID"])
    country = _clean_text(raw["Country"])
    quantity = pd.to_numeric(raw["Quantity"], errors="coerce")
    unit_price = pd.to_numeric(raw["UnitPrice"], errors="coerce")
    parsed_date = pd.to_datetime(raw["InvoiceDate"], errors="coerce")

    is_cancellation = invoice.str.upper().str.startswith("C", na=False)
    is_return_like = is_cancellation | quantity.lt(0)

    # A CustomerID is usable only when it is a 5-digit code; anything else is
    # stored as NULL with has_customer_id = 0 (the row itself is kept).
    customer_ok = customer.str.fullmatch(CUSTOMER_PATTERN, na=False)
    customer_id = customer.where(customer_ok)

    checks = {
        "missing_invoice_id": invoice.isna(),
        "invalid_invoice_id": invoice.notna()
        & ~invoice.str.fullmatch(INVOICE_PATTERN, na=False),
        "invalid_timestamp": parsed_date.isna(),
        "invalid_quantity_value": quantity.isna(),
        "invalid_unit_price_value": unit_price.isna(),
        "missing_stock_code": stock_code.isna(),
        # Compare source columns only: source_row_number is unique per row and
        # would hide every real duplicate. The first copy is kept.
        "exact_duplicate_extra_copy": raw.duplicated(
            subset=list(EXPECTED_COLUMNS), keep="first"
        ),
        "non_product_line": stock_code.str.upper().isin(NON_PRODUCT_STOCK_CODES)
        | stock_code.str.upper().str.startswith(NON_PRODUCT_PREFIXES, na=False),
        "nonpositive_unit_price": unit_price.le(0),
        "zero_quantity": quantity.eq(0),
        "cancellation_with_positive_quantity": is_cancellation & quantity.gt(0),
    }
    masks = pd.DataFrame(
        {name: pd.Series(mask).fillna(False).astype(bool).to_numpy() for name, mask in checks.items()},
        index=raw.index,
    )[list(EXCLUSION_RULES)]

    is_excluded = masks.any(axis=1)
    reasons = masks.apply(lambda row: "|".join(row.index[row.to_numpy()]), axis=1)
    primary = masks.idxmax(axis=1).where(is_excluded)

    iso_datetime = parsed_date.dt.strftime("%Y-%m-%d %H:%M:%S")
    iso_date = parsed_date.dt.strftime("%Y-%m-%d")
    line_value = (quantity * unit_price).round(4)

    base = pd.DataFrame(
        {
            "source_row_number": raw["source_row_number"],
            "invoice_no": invoice,
            "stock_code": stock_code,
            "description": description,
            "quantity": quantity,
            "unit_price": unit_price,
            "invoice_datetime": iso_datetime,
            "invoice_date": iso_date,
            "customer_id": customer_id,
            "has_customer_id": customer_ok.astype(int),
            "country": country,
        }
    )

    valid = ~is_excluded
    sales = base.loc[valid & ~is_return_like].copy()
    sales["quantity"] = sales["quantity"].astype("int64")
    sales["revenue"] = line_value.loc[sales.index]
    sales = sales[
        [
            "source_row_number", "invoice_no", "stock_code", "description",
            "quantity", "unit_price", "revenue", "invoice_datetime",
            "invoice_date", "customer_id", "has_customer_id", "country",
        ]
    ]

    returns = base.loc[valid & is_return_like].copy()
    returns["quantity"] = returns["quantity"].astype("int64")
    # Positive money value of the returned goods (quantity is negative).
    returns["return_value"] = (-line_value.loc[returns.index]).round(4)
    returns = returns[
        [
            "source_row_number", "invoice_no", "stock_code", "description",
            "quantity", "unit_price", "return_value", "invoice_datetime",
            "invoice_date", "customer_id", "has_customer_id", "country",
        ]
    ]

    excluded = base.loc[is_excluded, [
        "source_row_number", "invoice_no", "stock_code", "description",
        "quantity", "unit_price", "invoice_datetime", "customer_id", "country",
    ]].copy()
    excluded["exclusion_reasons"] = reasons.loc[excluded.index]
    excluded["primary_reason"] = primary.loc[excluded.index]

    primary_counts = {
        rule: int(primary.eq(rule).sum()) for rule in EXCLUSION_RULES
    }
    no_id = ~customer_ok
    return CleanResult(
        sales=sales.reset_index(drop=True),
        returns=returns.reset_index(drop=True),
        excluded=excluded.reset_index(drop=True),
        rule_counts={rule: int(masks[rule].sum()) for rule in EXCLUSION_RULES},
        primary_counts=primary_counts,
        no_customer_id_counts={
            "source_rows": int(no_id.sum()),
            "sales": int((sales["has_customer_id"] == 0).sum()),
            "returns": int((returns["has_customer_id"] == 0).sum()),
            "excluded": int(no_id[is_excluded].sum()),
        },
        invalid_customer_id_format=int((customer.notna() & ~customer_ok).sum()),
    )
