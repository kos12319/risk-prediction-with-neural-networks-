# ADR 0101 — Random K‑Fold CV for Temporal Data

- Status: Rejected
- Date: 2025-09-18

## Context
Random K‑fold cross‑validation mixes future and past observations across folds for time‑ordered data.

## Decision
Do not use random K‑fold CV for temporal data; prefer time‑aware splits.

## Rationale
- Introduces temporal leakage; inflates performance.

## Alternatives Considered
- Forward‑chaining temporal CV: appropriate and proposed elsewhere.

