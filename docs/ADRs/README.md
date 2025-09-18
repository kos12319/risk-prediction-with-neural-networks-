# Architecture Decision Records (ADRs)

This directory tracks architectural decisions, organized by status:

- accepted/ — decisions adopted by the project
- proposed/ — proposals under consideration
- rejected/ — proposals considered but not adopted

Conventions:
- Use incremental IDs with zero padding (e.g., 0001-...).
- Each ADR includes: Context, Decision (or Proposal), Rationale, Consequences, Alternatives, and Status.

Current ADRs:
- accepted/0001-time-based-split.md — Use time-based split for evaluation (Accepted)
- accepted/0003-backend-pytorch.md — Select PyTorch as the training/inference backend (Accepted)
- accepted/0004-threshold-on-validation.md — Choose threshold on validation; apply to test (Accepted)
- accepted/0005-oversampling-isolation-class-weights.md — Oversample train subset only; support class weights (Accepted)
- accepted/0006-leakage-controls-origination.md — Drop post‑origination features by default (Accepted)
- accepted/0007-single-run-folder-artifacts.md — Single per‑run folder under local_runs/ (Accepted)
- accepted/0008-dependency-management-pip-tools.md — Manage deps with pip‑tools (Accepted)
- proposed/0002-temporal-cv-for-selection.md — Add temporal cross-validation for feature selection/tuning (Proposed)
- proposed/0009-calibration-post-training.md — Optional calibration (Platt/Isotonic) on validation (Proposed)
- proposed/0010-feature-selection-policy-stability.md — Ensemble MI+L1 ranking with stability criteria (Proposed)
- proposed/0011-positive-class-convention.md — Standardize default positive class convention (Proposed)
- proposed/0012-run-ledger.md — Run ledger (experiments.csv) for reproducibility (Proposed)
- rejected/0101-random-kfold-temporal.md — Random K‑fold CV for temporal data (Rejected)
- rejected/0102-random-split-default.md — Random split as default (Rejected)
- rejected/0103-include-pricing-fields-in-agnostic.md — Include pricing/scoring fields in agnostic model (Rejected)
- rejected/0104-oversample-val-test.md — Oversampling validation/test splits (Rejected)
- rejected/0105-store-large-data-and-runs-in-git.md — Store large data and runs in Git (Rejected)
- rejected/0106-force-dense-everywhere.md — Force dense representations end‑to‑end (Rejected)
