# Run Artifacts and Reproducibility

Every training workflow writes into `local_runs/` so that model state, metrics, and provenance stay reproducible. This note reflects the current pipeline behaviour (PyTorch backend; H2O follows the same layout for shared files).

## Directory Layout
- Runs live under `local_runs/<group>/run_YYYYMMDD_HHMMSS/` when `output.runs_root` is set (default). The `<group>` token is derived from dataset stem, split method, positive-class token (`co` = Charged Off, `fp` = Fully Paid), and backend (e.g. `thesis_data_sample_1k|time|co|pytorch`).
- Within a run directory the pipeline writes everything directly at the top level, with plots nested under `figures/`. The older `reports/`/`models/` hierarchy is no longer used unless `output.models_dir`/`output.reports_dir` are customised in a config.

## Standard Training Run Contents
- `config_resolved.yaml` – fully expanded config after processing `extends`.
- `metrics.json` – ROC AUC, Average Precision, thresholded classification report, and summary scalars.
- `confusion.json` – TP/FP/TN/FN along with precision/recall/FPR at the chosen threshold.
- `loan_default_model.pt` – serialised PyTorch weights (filename controlled by `output.model_filename`).
- `features.json` – numerical/categorical feature lists and the encoded feature names emitted by the preprocessor.
- `data_manifest.json` – dataset provenance: CSV path, SHA256, row/column counts, class counts, time ranges for dataset/train/val/test, optional validation counts, and a `resampling` block when oversampling ran.
- `roc_points.csv`, `pr_points.csv` – per-threshold ROC/PR sweeps.
- `threshold_metrics.csv` – precision/recall/TPR/FPR/specificity/F1 evaluated on a 0.00–1.00 grid.
- `requirements.freeze.txt` – exact Python environment captured via `pip freeze` for portability.
- `figures/learning_curves.png`, `figures/roc_curve.png`, `figures/pr_curve.png` – PNG plots with the selected operating point annotated.

Notes:
- The single-run pipeline no longer emits a local `README.md`; the rich summary is attached to the W&B artifact when tracking is enabled. Temporal CV runs (see below) still create a summary README at the run root.
- A `wandb.json` helper file appears when W&B tracking succeeds (contains run id, path, and URL). When network access or permissions prevent W&B from launching this file is omitted.

## Temporal CV Additions (`split.cv.enabled`)
- `cv_metrics.json` – aggregated fold metrics plus per-fold confusion counts, thresholds, and duration stats.
- `README.md` – concise textual summary of mean/std metrics across folds (always written even when W&B is disabled).
- `folds/fold_XX/` – one directory per fold storing `metrics.json`, `confusion.json`, and the fold-specific model file (named `loan_default_model_foldXX.<ext>`). Fold plots are staged under `figures/folds/fold_XX/` when generated.
- If `train_full_after: true`, the run directory also contains the standard single-run artifacts for the refit model (metrics, confusion, curves, manifest, etc.).

## H2O AutoML Extras
- Models serialise to `.zip` (`loan_default_model.zip` by default); fold models adopt the `_foldXX.zip` suffix.
- `h2o_leaderboard.csv`/`.json` – master leaderboard with validation metrics; optional `h2o_leaderboard_test.csv`/`h2o_leaderboard_validation.csv` appear when `leaderboard_make_test` or similar flags are enabled.
- `h2o_leaderboard_roc.png`, `h2o_leaderboard_pr.png`, `h2o_pareto_front.png`, `h2o_model_correlation.png` – comparison plots under `figures/comparison/` (top models, per-family winners).
- Feature insights: `varimp_per_family/varimp_<algo>.csv` with PNGs under `figures/comparison/per_family_varimp/`, and optional `partial_dependence/partial_<feature>.csv` with plots inside `figures/explanations/partial_dependence/`.
- Logs and diagnostics: `h2o_logs/` captures the embedded H2O server output (debug/info) for audit trails.
- When leaderboard curve exports are requested, the pipeline saves ROC/PR raw points alongside the PNGs for replotting.

## Example `data_manifest.json`
```json
{
  "csv_path": "data/raw/samples/thesis_data_sample_1k.csv",
  "csv_path_abs": "/Users/petros/Projects/risk-prediction-with-neural-networks-/data/raw/samples/thesis_data_sample_1k.csv",
  "filesize_bytes": 751523,
  "mtime": 1758094672,
  "sha256": "57cf2e1886845c64ad0ddede56e2a19f0e88d73f6b15d7615f7ba3a4e0e07e45",
  "n_rows": 873,
  "n_cols": 44,
  "class_counts": {"1": 728, "0": 145},
  "date_ranges": {
    "dataset": {"min": "2015-12-01", "max": "2015-12-01"},
    "train": {"min": "2015-12-01", "max": "2015-12-01"},
    "val": {"min": "2015-12-01", "max": "2015-12-01"},
    "test": {"min": "2015-12-01", "max": "2015-12-01"}
  },
  "train_class_counts": {"0": 467, "1": 467},
  "test_class_counts": {"0": 26, "1": 149},
  "val_class_counts": {"0": 28, "1": 112},
  "resampling": {
    "method": "random_over_sampler",
    "before_counts": {"0": 91, "1": 467},
    "after_counts": {"0": 467, "1": 467}
  }
}
```

## Catalog Helpers
- `make run-catalog RUNS_ROOT=local_runs` scans all `run_*` folders and builds `local_runs/_catalog.json` summarising metrics, confusion, manifest, model filenames, and available figures.
- `make run-catalog-report RUNS_ROOT=local_runs` renders `local_runs/index.md` plus small trend plots under `local_runs/index_plots/` for quick comparison across runs.

These commands rely on the artifact layout described above; keeping the structure stable ensures reports stay correct.
