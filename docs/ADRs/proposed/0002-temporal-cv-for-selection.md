# ADR 0002 — Add Temporal Cross‑Validation for Feature Selection and Tuning

- Status: Proposed
- Date: 2025-09-18

## Context
Current feature selection (MI/L1) and tuning use a single time‑based train/test split. While time‑aware, a single cut produces high‑variance estimates and can be sensitive to where the split falls relative to macro or policy shifts.

## Proposal
Introduce temporal cross‑validation (forward‑chaining) to estimate performance of feature subsets and tuning choices across multiple time windows, then aggregate (mean and confidence bands). Keep a final out‑of‑time test set for reporting.

## Rationale
- Stability: Multiple temporal folds reduce variance vs a single holdout and help select a subset that generalizes across time.
- Robustness to drift: Highlights features whose utility is unstable across cohorts.
- Better operating points: Thresholds chosen on per‑fold validation are less overfit to a single period.

## Consequences
- More compute and runtime; added complexity in selection CLI and reporting.
- Clearer, more credible curves with mean±CI across folds; improved decision‑making for subset size.
- Must preserve invariants: no leakage; oversample train subset only; seed splits; respect configured positive class.

## Alternatives Considered
- Single time holdout (status quo): simple and fast but higher variance and sensitive to split choice.
- Random K‑fold CV: inappropriate for temporal data due to leakage across folds.

## Implementation Sketch
- CLI: `--cv_folds N --cv_mode expanding|rolling --cv_val_size 0.2`.
- Splitter: generate time‑ordered folds; for each fold, split train→(train_sub, val_sub); fit preprocessor on train_sub; optional ROS on train_sub only (seeded).
- Selection loop: compute ranking on train_sub; evaluate incremental subsets on val_sub (or fold test) and aggregate AUC across folds; stop at target coverage.
- Output: write per‑fold metrics, aggregate curves (mean/bands), and a stability score per feature (e.g., selection frequency).

