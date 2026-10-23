"""Write the cleaned tables into a fresh SQLite file."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .cleaning import CleanResult

SCHEMA = """
CREATE TABLE sales (
    source_row_number INTEGER PRIMARY KEY,
    invoice_no        TEXT    NOT NULL,
    stock_code        TEXT    NOT NULL,
    description       TEXT,
    quantity          INTEGER NOT NULL CHECK (quantity > 0),
    unit_price        REAL    NOT NULL CHECK (unit_price > 0),
    revenue           REAL    NOT NULL,
    invoice_datetime  TEXT    NOT NULL,
    invoice_date      TEXT    NOT NULL,
    customer_id       TEXT,
    has_customer_id   INTEGER NOT NULL CHECK (has_customer_id IN (0, 1)),
    country           TEXT
);
CREATE TABLE returns (
    source_row_number INTEGER PRIMARY KEY,
    invoice_no        TEXT    NOT NULL,
    stock_code        TEXT    NOT NULL,
    description       TEXT,
    quantity          INTEGER NOT NULL CHECK (quantity < 0),
    unit_price        REAL    NOT NULL CHECK (unit_price > 0),
    return_value      REAL    NOT NULL,
    invoice_datetime  TEXT    NOT NULL,
    invoice_date      TEXT    NOT NULL,
    customer_id       TEXT,
    has_customer_id   INTEGER NOT NULL CHECK (has_customer_id IN (0, 1)),
    country           TEXT
);
CREATE TABLE excluded_rows (
    source_row_number INTEGER PRIMARY KEY,
    invoice_no        TEXT,
    stock_code        TEXT,
    description       TEXT,
    quantity          REAL,
    unit_price        REAL,
    invoice_datetime  TEXT,
    customer_id       TEXT,
    country           TEXT,
    exclusion_reasons TEXT NOT NULL,
    primary_reason    TEXT NOT NULL
);
CREATE INDEX idx_sales_date ON sales (invoice_date);
CREATE INDEX idx_sales_customer ON sales (customer_id);
CREATE INDEX idx_sales_stock ON sales (stock_code);
CREATE INDEX idx_sales_country ON sales (country);
CREATE INDEX idx_returns_date ON returns (invoice_date);
"""


def _rows(frame: pd.DataFrame):
    """Yield tuples with NaN/NA turned into None so SQLite stores NULL."""
    clean = frame.astype(object).where(frame.notna(), None)
    return clean.itertuples(index=False, name=None)


def write_database(result: CleanResult, db_path: Path) -> None:
    """Create ``db_path`` from scratch (an existing file is replaced)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(db_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)

    connection = sqlite3.connect(tmp_path)
    try:
        connection.executescript(SCHEMA)
        for name, frame in (
            ("sales", result.sales),
            ("returns", result.returns),
            ("excluded_rows", result.excluded),
        ):
            placeholders = ", ".join("?" * len(frame.columns))
            connection.executemany(
                f"INSERT INTO {name} ({', '.join(frame.columns)}) VALUES ({placeholders})",
                _rows(frame),
            )
        connection.commit()
    finally:
        connection.close()
    tmp_path.replace(db_path)
