# ADR 0104 — Oversample Validation/Test Splits

- Status: Rejected
- Date: 2025-09-18

## Context
Oversampling non‑training splits inflates signal and biases evaluation.

## Decision
Never oversample validation or test splits; resampling is restricted to the training subset only.

## Rationale
- Preserves unbiased validation and test metrics; avoids leakage.

## Alternatives Considered
- Weighting or focal loss: acceptable, applied during training only.

