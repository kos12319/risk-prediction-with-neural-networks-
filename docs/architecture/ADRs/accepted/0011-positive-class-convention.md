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
- Code: probability alignment and metric computation live in `src/training/base_pipeline.py`; backend adapters (e.g., `src/training/backends/pytorch/pipeline.py`, `src/training/backends/h2o/pipeline.py`) feed backend-specific outputs while respecting the configured `pos_label`.
- CLI/Explore: `src/cli/explore.py` normalizes string values (“default”, “charged off”) to 0.
