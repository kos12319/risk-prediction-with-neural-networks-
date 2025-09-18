# ADR 0011 — Positive Class Convention

- Status: Proposed
- Date: 2025-09-18

## Context
The project can report metrics/curves with either class as positive. In credit risk, treating defaults as positive is often useful for recall and PR interpretation.

## Proposal
Standardize default `eval.pos_label=0` (Charged Off) with explicit inversion logic in evaluation; document impacts on curves/metrics and make the setting prominent in run summaries.

## Rationale
- Aligns metrics with risk‑detection goals.
- Eliminates confusion when comparing runs.

## Consequences
- No behavior change for users who override pos_label.

## Alternatives Considered
- Default to fully paid: acceptable but less aligned with default‑detection framing.

