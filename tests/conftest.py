import sqlite3

import pandas as pd
import pytest

from src.retail_analytics.cleaning import clean_transactions
from src.retail_analytics.database import write_database


def _row(invoice, stock, qty, price, customer, country="United Kingdom", date="2011-01-05 10:00:00"):
    return {
        "InvoiceNo": invoice, "StockCode": stock, "Description": f"ITEM {stock}",
        "Quantity": qty, "InvoiceDate": date, "UnitPrice": price,
        "CustomerID": customer, "Country": country,
    }


@pytest.fixture(scope="session")
def retail_db(tmp_path_factory):
    """Small database with the real schema: sales, returns and excluded_rows."""
    rows = [
        _row("536365", "85123A", 6, 2.55, "17850"),
        _row("536366", "71053", 3, 3.39, "17850", date="2011-02-07 09:00:00"),
        _row("536367", "22752", 2, 7.65, "13047", "France", "2011-02-08 11:30:00"),
        _row("536368", "21730", 4, 4.25, None, "Germany", "2011-03-09 12:00:00"),
        _row("C536379", "22423", -1, 12.5, "14527"),
        _row("536400", "POST", 1, 18.0, "14527"),
        _row("536401", "21730", 1, 0.0, "14527"),
    ]
    frame = pd.DataFrame(rows)
    for col in ("InvoiceNo", "StockCode", "Description", "CustomerID", "Country"):
        frame[col] = frame[col].astype("string")
    frame.insert(0, "source_row_number", range(2, len(frame) + 2))
    path = tmp_path_factory.mktemp("db") / "retail.sqlite"
    write_database(clean_transactions(frame), path)
    # Extra rows so a 3-way self join is slow enough to hit the timeout test.
    with sqlite3.connect(path) as con:
        con.executemany(
            "INSERT INTO sales (source_row_number, invoice_no, stock_code, quantity, unit_price,"
            " revenue, invoice_datetime, invoice_date, has_customer_id)"
            " VALUES (?, 'B', 'X', 1, 1.0, 1.0, '2011-01-01 00:00:00', '2011-01-01', 0)",
            [(1000 + i,) for i in range(3000)],
        )
    return path
