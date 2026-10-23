"""Read the UCI Online Retail workbook without changing the source file.

Adapted from the Project 1 ingestion step (my-new-project/src/data/ingestion.py).
"""

from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = (
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
)


def load_transactions(source_path: Path) -> pd.DataFrame:
    """Load the workbook and attach its original Excel row number.

    Identifiers are read as strings so numeric-looking values keep their
    identifier form. Source cells are otherwise left intact.
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source workbook not found: {source_path}")
    if source_path.suffix.lower() != ".xlsx":
        raise ValueError(f"Expected an .xlsx workbook, got: {source_path.name}")

    frame = pd.read_excel(
        source_path,
        engine="openpyxl",
        dtype={
            "InvoiceNo": "string",
            "StockCode": "string",
            "Description": "string",
            "CustomerID": "string",
            "Country": "string",
        },
    )
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "Workbook is missing expected UCI columns: " + ", ".join(missing_columns)
        )

    # Excel row 1 is the header, so the first transaction is row 2.
    frame.insert(0, "source_row_number", range(2, len(frame) + 2))
    return frame
