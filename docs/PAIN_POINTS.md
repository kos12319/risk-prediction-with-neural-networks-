# Pain Points and Recommendations

This document captures friction points identified while reviewing the repo, with quick recommendations. Updated to reflect current code and scope.

Legend: [Status: Addressed | Partially addressed | Still applies]

-## Reproducibility & Environment
- Unpinned dependencies. [Status: Addressed] `requirements.txt` pins versions. Maintain the pip‑tools workflow to keep the lock consistent.
- PyTorch install friction. [Status: Partially addressed] Pinning helps and README notes CPU‑only guidance; an optional CPU constraints file could further reduce friction.
- Missing global seeding. [Status: Addressed] Added `set_seed`, seeded Torch splits and DataLoader workers.
- requirements.in discoverability. [Status: Addressed] There was confusion about a missing `requirements.in`. The repo includes it at the root and the README documents pip‑tools. Recommendation: reference `requirements.in` prominently in contribution guidelines and ensure devs use `make deps-compile`/`make deps-sync`.

## Config & Data Loading
- Date columns not guaranteed to load. [Status: Addressed] `load_and_prepare()` now includes `parse_dates` in `usecols`.
- Selection CLI ignores `extends`. [Status: Addressed] Selection resolves `extends` like training.

## Modeling & Training
- PyTorch validation split nondeterminism. [Status: Addressed] Use a seeded `torch.Generator` and worker seeding.
- Sparse→dense conversion via `.todense()`. [Status: Addressed] Switched to `.toarray()` with `float32` downstream.
- `time_based_split` docstring vs behavior. [Status: Addressed] Docstring clarified to reflect index split after sorting.
- Oversampling before validation split. [Status: Addressed] Now splits train→(train, val) before ROS; preprocessor fits on train subset only.

## Evaluation & Reporting
- Unconditional positive‑class note. [Status: Addressed] README note reflects the configured `eval.pos_label`.
- Absolute paths in run README. [Status: Addressed] Artifact paths in README are relative to the run dir.
- Threshold sanity. [Status: Addressed] Adds a note when precision/recall are near 0 at the chosen threshold.
- Threshold chosen on the test set. [Status: Addressed] Threshold is chosen on validation (fallback to test if no val) and applied to test metrics.
- W&B threshold sweep plots show one point. [Status: Addressed] Full ROC/PR sweeps are logged as W&B tables/plots.

## Feature Selection
- Fragile OHE group mapping. [Status: Addressed] Grouping uses encoder introspection with `categories_` and ColumnTransformer structure.
- Single holdout evaluation. [Status: Still applies] Consider temporal CV (rolling windows) or repeated holdout for more stable AUC curves.

## Repo Hygiene & Tests
- Large data artifacts in git. [Status: Addressed] Archives are tracked via LFS and unzipped full datasets are gitignored.
- Stray/broken script. [Status: Addressed] `csv_prep.py` no longer exists.
- No tests. [Status: Partially addressed] Added minimal tests; expand to model I/O round‑trip, threshold selection correctness, and run artifacts schema.

## Minor/Edge Cases
- `credit_history_length` can be negative if dates are out of order; clamp to non‑negative or drop anomalies. [Status: Addressed]
- Oversampling documentation vs behavior. [Status: Addressed] README clarifies oversampling applies only to the training subset; class‑weight mode supported.
