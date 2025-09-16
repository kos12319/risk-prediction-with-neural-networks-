# Run Artifacts and Reproducibility

Each training run creates a timestamped folder under `reports/runs/run_YYYYMMDD_HHMMSS/` that captures all key artifacts for analysis and reproducibility.

## Contents
- README.md — human summary of config, backend, positive class, threshold strategy, key metrics, confusion stats, dataset/model info.
- metrics.json — ROC AUC, Average Precision, classification report at the chosen threshold.
- confusion.json — TP/FP/TN/FN, precision/recall/specificity at the chosen threshold.
- figures/
  - learning_curves.png — train/val loss by epoch.
  - roc_curve.png — ROC with chosen operating point annotated.
  - pr_curve.png — PR with chosen operating point annotated.
- history.csv — epoch, loss, val_loss table.
- roc_points.csv — threshold sweep points for ROC (`threshold,fpr,tpr`).
- pr_points.csv — threshold sweep points for PR (`threshold,precision,recall`).
- config_resolved.yaml — fully resolved configuration for the run (after `extends`).
- features.json — numerical/categorical feature lists and the final `feature_inputs` used.
- data_manifest.json — dataset provenance and cohort ranges (new):
  - csv_path (relative and absolute)
  - filesize_bytes, mtime, sha256
  - n_rows, n_cols, overall class_counts
  - date_ranges: dataset/train/test min/max for `issue_d` when available
  - train_class_counts, test_class_counts
- <model file> — a copy of the trained model in the run folder (e.g., `loan_default_model.pt`).

## Why this matters
- Reprovenance: Ties metrics to the exact dataset and configuration used.
- Temporal clarity: Records the train/test date windows under time-based splits.
- Auditability: Enables consistent tables and figures across runs for the thesis.

See also: `docs/thesis/PIVOT_SUGGESTIONS.md` for a broader plan on experiment tracking and provenance.

## Example `data_manifest.json`
The values below are illustrative.

```
{
  "csv_path": "data/raw/samples/first_10k_rows.csv",
  "csv_path_abs": "/abs/path/to/project/data/raw/samples/first_10k_rows.csv",
  "filesize_bytes": 7490945,
  "mtime": 1726440000,
  "sha256": "2a1c7d...deadbeef",
  "n_rows": 10000,
  "n_cols": 120,
  "class_counts": {"0": 8000, "1": 2000},
  "date_ranges": {
    "dataset": {"min": "2015-01-01", "max": "2017-12-01"},
    "train": {"min": "2015-01-01", "max": "2017-05-01"},
    "test": {"min": "2017-05-02", "max": "2017-12-01"}
  },
  "train_class_counts": {"0": 6400, "1": 1600},
  "test_class_counts": {"0": 1600, "1": 400}
}
```
