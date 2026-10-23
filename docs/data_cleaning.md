# Data cleaning

> **Status:** built and reconciled against Project 1 on 2010-12-01..2011-12-09
> data (source SHA-256 `43465a06...6676d`). Counts below are real output of
> `scripts/build_db.py`. The non-product list was extended after review (see
> "Stock-code review").

## Source

UCI Online Retail dataset (UK online gift retailer, 2010-12-01 to 2011-12-09).
File: `data/raw/Online Retail.xlsx` (git-ignored). Download page:
<https://archive.ics.uci.edu/dataset/352/online+retail>
(direct zip used by Project 1: `https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip`).

Rebuild: `env -u PYTHONPATH .venv/bin/python scripts/build_db.py`
Output: `data/processed/retail.sqlite` (git-ignored) and `build_report.json`.

## Principle

Every source row lands in exactly one of three tables. Nothing is silently
deleted: `sales + returns + excluded_rows = source rows`. The build script
stops with an error if that does not hold.

## Tables

| Table | Contents |
|---|---|
| `sales` | Valid product sales lines (positive quantity and price). |
| `returns` | Valid cancellation / return lines (negative quantity). |
| `excluded_rows` | Rows that broke a rule, with `exclusion_reasons` (all, `\|`-joined) and `primary_reason` (first rule in the priority order below). |

Key columns: `invoice_datetime` (`YYYY-MM-DD HH:MM:SS`), `invoice_date`
(`YYYY-MM-DD`), `customer_id` (text, NULL if unknown), `has_customer_id`
(0/1), `revenue` (sales) / `return_value` (returns).

## Exclusion rules (priority order)

| # | Rule | Row is excluded when | Why |
|---|---|---|---|
| 1 | `missing_invoice_id` | InvoiceNo empty | Cannot identify the order. |
| 2 | `invalid_invoice_id` | InvoiceNo is not 6 digits with optional leading `C` (e.g. `A563185`) | Non-order records such as bad-debt adjustments. |
| 3 | `invalid_timestamp` | InvoiceDate cannot be parsed | Cannot place it in time. |
| 4 | `invalid_quantity_value` | Quantity not numeric | Cannot compute revenue. |
| 5 | `invalid_unit_price_value` | UnitPrice not numeric | Cannot compute revenue. |
| 6 | `missing_stock_code` | StockCode empty | Not a product line. |
| 7 | `exact_duplicate_extra_copy` | All 8 source columns equal an earlier row (first copy kept) | Likely double entry; counted once. |
| 8 | `non_product_line` | StockCode (case-insensitive) is one of `POST, DOT, M, C2, D, S, B, BANK CHARGES, AMAZONFEE, CRUK` (Project 1 list), or `23444` (Next Day Carriage), `23574` (PACKING CHARGE), `22016` (GBP 100 gift voucher), or starts with `GIFT_` (gift vouchers) | Postage, manual, carriage, fees, discounts, samples and prepaid vouchers are not products sold. |
| 9 | `nonpositive_unit_price` | UnitPrice <= 0 | Free/adjustment lines, not real sales. |
| 10 | `zero_quantity` | Quantity = 0 | Neither a sale nor a return. |
| 11 | `cancellation_with_positive_quantity` | InvoiceNo starts with `C` but Quantity > 0 | Contradictory record. |

A row can break several rules; `exclusion_reasons` lists all of them.
Rule counts therefore overlap, while `primary_reason` counts add up to the
total number of excluded rows.

**Not a rule:** a missing or malformed CustomerID never excludes a row. The
row is kept, `customer_id` is NULL and `has_customer_id = 0`.
Customer-level analysis must filter on `has_customer_id = 1`.

## Routing of rows that pass every rule

* InvoiceNo starts with `C` **or** Quantity < 0 -> `returns`
* otherwise (Quantity > 0) -> `sales`

## Revenue definitions

* **Line revenue** (`sales.revenue`) = `Quantity * UnitPrice`, in GBP, rounded to 4 decimals.
* **Gross revenue** = `SUM(sales.revenue)`. Returns are *not* subtracted.
  This is the default meaning of "revenue" / "sales" for the assistant.
* **Return value** (`returns.return_value`) = `ABS(Quantity) * UnitPrice`, a positive amount.
* **Net revenue** = gross revenue - `SUM(returns.return_value)`.
  Only on request ("net of returns").
* Returns cannot be matched to the original sale (no order link in the
  dataset), so a return is subtracted in the period it was *issued*, not the
  period of the original sale.
* Includes rows without a CustomerID (guest/unknown buyers). Project 1
  excluded them because it analysed customers.

## Real row counts

Source: 541,909 rows.

| Table | Rows | Revenue / value (GBP) |
|---|---:|---:|
| `sales` | 522,446 | 10,245,830.99 gross revenue |
| `returns` | 8,665 | 475,871.16 return value |
| `excluded_rows` | 10,798 | not valued |
| **Total** | **541,909** | |

Net revenue = 10,245,830.99 - 475,871.16 = **9,769,959.83**.
Sales span 2010-12-01 to 2011-12-09.

### `excluded_rows` by reason

| Rule | Rows breaking the rule (overlapping) | As primary reason (adds to 10,798) |
|---|---:|---:|
| `exact_duplicate_extra_copy` | 5,268 | 5,268 |
| `non_product_line` | 3,045 | 3,036 |
| `nonpositive_unit_price` | 2,517 | 2,491 |
| `invalid_invoice_id` | 3 | 3 |
| all other rules (missing invoice, bad timestamp/number, missing stock code, zero quantity, cancellation with positive quantity) | 0 | 0 |
| **Total** | | **10,798** |

Combinations: 3,017 non-product only; 19 non-product + non-positive price;
6 duplicate + non-product; 5 duplicate + non-positive price; 2 invalid invoice
+ non-product + non-positive price; 1 invalid invoice + non-product.
Of the 10,798 excluded rows, 3,532 have no CustomerID and 7,266 have one.

### Rows without a CustomerID

135,080 source rows have no CustomerID: 131,386 are in `sales`, 162 in
`returns`, 3,532 in `excluded_rows` (131,386 + 162 + 3,532 = 135,080).
No CustomerID was malformed (0 non-empty IDs failed the 5-digit check).

## Reconciliation to Project 1

Project 1 (`online_retail.sqlite`) kept 541,909 audited rows, of which
391,153 were "customer sales candidates" with gross revenue 8,737,227.64
(`Quantity * UnitPrice`). All comparisons were checked row by row on
`source_row_number` (Project 1 database opened read-only).

### Bridge: Project 1 candidates -> new `sales`

| Step | Rows | Revenue (GBP) |
|---|---:|---:|
| Project 1 customer sales candidates (all have a CustomerID) | 391,153 | 8,737,227.64 |
| - `23444` Next Day Carriage (now non-product) | -79 | -1,200.00 |
| - `23574` PACKING CHARGE (now non-product) | -14 | -105.00 |
| (no gift voucher row was a Project 1 candidate: all 32 have no CustomerID) | 0 | 0.00 |
| Any other rule difference (all other rules identical) | 0 | 0.00 |
| = new `sales` with `has_customer_id = 1` | 391,060 | 8,735,922.64 |
| + no-CustomerID rows, now kept (131,418 before the list change) | +131,418 | +1,510,677.49 |
| - gift vouchers (`gift_*`, `22016`) among them, now non-product | -32 | -769.14 |
| = no-CustomerID rows in `sales` | 131,386 | 1,509,908.35 |
| **= new `sales`** | **522,446** | **10,245,830.99** |

* The voucher/fee change removes 125 rows and 2,074.14 in total
  (93 rows / 1,305.00 with a CustomerID + 32 rows / 769.14 without).
* Revenue differs from 8,737,227.64 for two reasons: +1,509,908.35 from kept
  no-ID rows, and -1,305.00 from the two fee codes. Without the list change it
  would differ by +1,510,677.49 only.
* Every other row in Project 1's candidate set is in `sales`, and every
  `sales` row with an ID was a Project 1 candidate.

### Bridge: Project 1 non-candidates (150,756) -> new tables

| Step | Rows |
|---|---:|
| Project 1 non-candidates | 150,756 |
| + candidates moved out by the new non-product list (`23444`, `23574`) | +93 |
| = rows that are not an ID-bearing sale in the new database | 150,849 |

Where those 150,849 rows now are:

| Destination | Rows |
|---|---:|
| `sales` without CustomerID (kept now) | 131,386 |
| `returns` | 8,665 |
| `excluded_rows` | 10,798 |
| **Total** | **150,849** |

`returns` = Project 1's 8,506 net-spend cancellation lines minus 3 now-
non-product lines (`23444` x1, `23574` x2; value 30.00) = 8,503 with an ID
(471,720.71), plus 162 no-ID return lines (4,150.45) = 8,665 (475,871.16).
Project 1's 10,624 "cancellations/returns" rows are broader: they include
non-product, zero-price or duplicate lines, which now sit in `excluded_rows`.

Nothing failed to reconcile.

## Stock-code review

Non-numeric stock codes in the raw file (all rows):

| Code | Description | Rows | Handling |
|---|---|---:|---|
| POST | POSTAGE | 1,256 | non-product |
| DOT | DOTCOM POSTAGE | 710 | non-product |
| M / m | Manual | 572 | non-product |
| C2 | CARRIAGE | 144 | non-product |
| D | Discount | 77 | non-product |
| S | SAMPLES | 63 | non-product |
| BANK CHARGES | Bank Charges | 37 | non-product |
| AMAZONFEE | AMAZON FEE | 34 | non-product |
| CRUK | CRUK Commission | 16 | non-product |
| B | Adjust bad debt | 3 | non-product |
| gift_0001_10/20/30/40/50 | Dotcomgiftshop Gift Voucher GBP 10-50 | 34 | non-product (`GIFT_` prefix) |
| DCGSSBOY / DCGSSGIRL | party bags | 24 | **product**, kept |
| other DCGS* | eBay / dotcom shop items | 21 | **product**, kept |
| PADS | PADS TO MATCH ALL CUSHIONS (price 0.001) | 4 | **product**, kept |

Numeric codes with non-product descriptions, added to the list by exact code:
`22016` (Gift Voucher GBP 100), `23444` (Next Day Carriage),
`23574` (PACKING CHARGE). `23702` ("High Resolution Image", 3 rows, 27.00)
looks digital but is kept as a product.

Why vouchers are excluded: they are prepaid credit, not goods sold; counting
them as revenue would double count once the credit is spent on products.

## Known caveats: two huge cancelled orders (deliberately kept)

Both stay in the data (`sales` and `returns`):

| Order | Units | Value (GBP) | Date | Cancelled by |
|---|---:|---:|---|---|
| 541431 | 74,215 | 77,183.60 | 2011-01-18 | C541433 |
| 581483 | 80,995 | 168,469.60 | 2011-12-09 | C581484 |

Together 245,653.20 (about 2.4% of gross revenue). They cancel out in net
revenue but inflate gross revenue and monthly totals (January 2011 and
December 2011 in particular). Answers about gross revenue by period should
mention this when it matters.

Other caveats: returns are subtracted in the period issued, not when sold.
