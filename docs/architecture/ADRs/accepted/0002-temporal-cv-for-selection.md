# ADR 0002 — Add Temporal Cross‑Validation for Feature Selection and Tuning

- Status: Accepted
- Date: 2025-09-18

## Context
Single time‑based train/test splits yield high‑variance subset/tuning estimates and are sensitive to split placement relative to macro/policy shifts.

## Decision
Adopt temporal cross‑validation (forward‑chaining, expanding mode) for feature selection and hyperparameter tuning. Keep an untouched out‑of‑time test set for final reporting. Optionally refit on the full dataset after CV (`train_full_after: true`).

## Rationale
- Stabilizes subset/tuning decisions across vintages; reduces variance.
- Improves credibility of curves/operating points with aggregated metrics.
- Maintains leakage controls and positive‑class conventions per evaluation invariants.

## Consequences
- Added compute/runtime; introduces per‑fold artifacts and an aggregate report.
- Requires careful seeding and fold construction.

## Alternatives Considered
- Single time holdout: faster but higher variance; sensitive to split choice.
- Random K‑fold: rejected for temporal data due to leakage.

## Implementation Notes
- Config: `split.cv.enabled: true`, `n_folds: N`, `mode: expanding`, `validation_fraction`, `train_full_after` in `configs/*.yaml`.
- Code: temporal CV orchestration lives in `src/training/base_pipeline.py` (`_run_temporal_cv`), invoked via backend adapters under `src/training/backends/*/pipeline.py`.
- Artifacts: per‑fold runs under `local_runs/…/folds/fold_XX/`; aggregate `reports/cv_metrics.json` in the main run folder.
