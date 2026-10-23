# Evaluation: dev set

Model: openai/gpt-oss-120b  |  database: retail.sqlite  |  questions: 15 (scored 15)

| Metric | Value |
|---|---|
| Execution success (answerable) | 100.0% |
| Result correctness (answerable) | 100.0% |
| End-to-end correct (rows right and status ok) | 100.0% |
| False refusals (answerable refused) | 0.0% |
| Schema validity, first try | 100.0% |
| Number grounding, first draft | 100.0% |
| Number grounding, final | 100.0% |
| Refusal / clarification accuracy | 100.0% |
| Safety: attacks blocked | 100.0% (failures: none) |
| SQL retry rate (answerable) | 0.0% |
| Latency median / p95 (excl. rate-limit waits) | 1.934s / 3.419s |
| Database unchanged over the whole run | True |

## Rate limits (reported separately, never counted as wrong answers)

- retries: 3, total time waiting: 93.0s
- questions that stayed rate-limited and are NOT scored: none

## By category

| Category | Passed | Rate |
|---|---|---|
| ambiguous | 1/1 | 100.0% |
| customer_level | 2/2 | 100.0% |
| filtering | 4/4 | 100.0% |
| modify_data | 1/1 | 100.0% |
| period_comparison | 1/1 | 100.0% |
| prompt_injection | 1/1 | 100.0% |
| returns_net | 2/2 | 100.0% |
| simple_aggregation | 2/2 | 100.0% |
| unanswerable | 1/1 | 100.0% |

## Failures

