# ADR 0011 — Positive Class Convention and Metrics Alignment

- Status: Accepted
- Date: 2025-09-18

## Context
The dataset encodes labels as 0 = Charged Off and 1 = Fully Paid. Different libraries assume different default positive classes, leading to inconsistent metrics/curves unless explicitly aligned.

## Decision
Adopt `eval.pos_label: 0` (Charged Off) as the default positive class. Ensure probability alignment, threshold selection, ROC/PR curves, and confusion metrics consistently treat `pos_label` as the positive event. Allow override via config.

## Rationale
- Matches the project’s risk‑centric framing (defaults as “positives”).
- Avoids silent inversions of metrics and thresholds across tools/backends.

## Consequences
- All training/evaluation code must propagate `pos_label` explicitly.
- Reports and W&B charts label the positive class accordingly.

## Alternatives Considered
- Default to 1 = Fully Paid: inconsistent with risk framing; rejected.

## Implementation Notes
- Config: `configs/default.yaml:evalu.pos_label: 0` and variants under `configs/pytorch/*.yaml`.
- Code: `src/training/pipeline.py` aligns probability columns to `pos_label`, computes metrics/curves consistently; H2O path in `src/training/train_h2o.py` reorders factor levels to place the positive class last and aligns probability extraction.
- CLI/Explore: `src/cli/explore.py` normalizes string values (“default”, “charged off”) to 0.

