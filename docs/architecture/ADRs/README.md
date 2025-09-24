# Architecture Decision Records (ADRs)

This directory tracks architectural decisions, organized by status:

- accepted/ — decisions adopted by the project
- proposed/ — proposals under consideration
- rejected/ — proposals considered but not adopted

Conventions:
- Use incremental IDs with zero padding (e.g., 0001-...).
- Each ADR includes: Context, Decision (or Proposal), Rationale, Consequences, Alternatives, Status, and (when relevant) Implementation Notes/Links.

Index (high level):
- accepted/0001-time-based-split.md — Use time-based split for evaluation (Accepted)
- accepted/0002-temporal-cv-for-selection.md — Temporal CV for selection/tuning (Accepted)
- accepted/0003-backend-pytorch.md — Select PyTorch as the primary backend (Accepted)
- accepted/0004-threshold-on-validation.md — Choose threshold on validation; apply to test (Accepted)
- accepted/0005-oversampling-isolation-class-weights.md — Oversample train subset only; support class weights (Accepted)
- accepted/0006-leakage-controls-origination.md — Drop post‑origination features by default (Accepted)
- accepted/0007-single-run-folder-artifacts.md — Single per‑run folder under local_runs/ (Accepted)
- accepted/0008-dependency-management-pip-tools.md — Manage deps with pip‑tools (Accepted)
- accepted/0011-positive-class-convention.md — Default positive class is 0 (Charged Off); align metrics/curves/thresholding (Accepted)
- accepted/0013-makefile-first-policy.md — Makefile‑first workflow and safe env defaults (Accepted)
- accepted/0014-tracking-backend-wandb.md — Optional experiment tracking via W&B (Accepted)
- accepted/0015-backend-h2o-automl.md — H2O AutoML as comparative backend (Accepted)
- accepted/0016-provider-agnostic-default.md — Provider‑agnostic feature set default; provider‑aware optional (Accepted)
- proposed/0009-calibration-post-training.md — Optional calibration (Platt/Isotonic) on validation (Proposed)
- proposed/0010-feature-selection-policy-stability.md — Ensemble MI+L1 ranking with stability criteria (Proposed)
- proposed/0012-run-ledger.md — Run ledger (experiments.csv) for reproducibility (Proposed)
- rejected/... — as recorded

Note: ADRs moved from `docs/ADRs/` to `docs/architecture/ADRs/`. A legacy index remains at the old path for backward compatibility.

