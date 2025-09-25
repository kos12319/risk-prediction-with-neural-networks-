# Temporal CV accepted for selection/tuning

- Date: 2025-09-24
- Status: landed
- Tags: eval, cv, selection

## Summary
Adopted forward-chaining temporal cross-validation (expanding windows) for feature selection and hyperparameter tuning. Aggregates metrics across folds and optionally refits on full data.

## ADRs
- 0002 — see docs/architecture/ADRs/accepted/0002-temporal-cv-for-selection.md

## Impact
- Config: `split.cv.enabled: true`, `n_folds`, `mode: expanding`, `validation_fraction`, `train_full_after`.
- Code: CV orchestration and aggregation live in `src/training/base_pipeline.py` (`_run_temporal_cv`), invoked via backend adapters under `src/training/backends/*/pipeline.py`.
- Artifacts: `run_dir/folds/fold_XX/` and `reports/cv_metrics.json` in the main run directory.

## Next
- Evaluate rolling mode once implemented; document default fold counts for common dataset sizes.
