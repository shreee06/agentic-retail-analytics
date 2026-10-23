"""Check that every number in a written answer comes from the returned rows."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

NUMBER = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(%|bn|billion|million|thousand|m|k)?(?![\w])",
    re.IGNORECASE,
)
SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}


def _to_float(text: str) -> float | None:
    try:
        return float(Decimal(text.replace(",", "")))
    except InvalidOperation:
        return None


def numbers_in_text(text: str) -> list[float]:
    """All plain numbers in ``text`` (commas removed, no unit scaling)."""
    values = []
    for match in NUMBER.finditer(text):
        value = _to_float(match.group(1))
        if value is not None:
            values.append(value)
    return values


def allowed_numbers(question: str, columns: list[str], rows: list[list]) -> set[float]:
    """Numbers an answer may quote: row values, numbers inside text cells (dates,
    product names), numbers from the question, the row count and list ranks."""
    allowed: set[float] = {0.0, float(len(rows))}
    allowed.update(float(i) for i in range(1, len(rows) + 1))
    allowed.update(numbers_in_text(question))
    for row in rows:
        for cell in row:
            if isinstance(cell, bool) or cell is None:
                continue
            if isinstance(cell, (int, float)):
                allowed.add(float(cell))
            else:
                allowed.update(numbers_in_text(str(cell)))
    return allowed


def ungrounded_numbers(answer: str, allowed: set[float]) -> list[str]:
    """Numbers in ``answer`` that match nothing in ``allowed`` (as written, i.e.
    to the number of decimals shown, allowing k/m/bn/% scaling)."""
    bad = []
    for match in NUMBER.finditer(answer):
        raw, unit = match.group(1), (match.group(2) or "").lower()
        value = _to_float(raw)
        if value is None:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        scale = SCALES.get(unit, 1.0)
        half_unit = 0.5 * 10**-decimals
        # (value, tolerance) pairs the number may legitimately match
        options = [(value * scale, half_unit * scale + 1e-9)]
        if unit == "%":  # 12.5% may come from a 0.125 ratio
            options.append((value / 100, half_unit / 100 + 1e-9))
        if not any(abs(v - a) <= tol for v, tol in options for a in allowed):
            bad.append(match.group(0).strip())
    return bad
