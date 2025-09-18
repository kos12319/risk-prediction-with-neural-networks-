# ADR 0004 — Choose Threshold on Validation; Apply to Test

- Status: Accepted
- Date: 2025-09-18

## Context
Choosing an operating threshold using the test set leaks information and inflates reported performance.

## Decision
Select the threshold on a validation subset carved from the training period using the configured strategy (`fixed|youden_j|f1`), then apply that fixed threshold to the untouched test set for reporting.

## Rationale
- Avoids test leakage and optimistic bias.
- Produces more reliable operating metrics.

## Consequences
- Training orchestration must create a validation subset before any resampling.
- README and metrics files reflect validation‑selected thresholds.

## Alternatives Considered
- Pick on test: simpler but biased; rejected.

