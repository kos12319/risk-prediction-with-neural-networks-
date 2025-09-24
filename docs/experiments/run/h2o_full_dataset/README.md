# H2O AutoML — Full Dataset (Time Split)

## Context
- Dataset: `data/raw/full/thesis_data_full.csv` (LendingClub accepted loans 2007–2018).
- Split policy: chronological by `issue_d` with validation carved from the training period; test uses the most recent vintages.
- Positive class: `Charged Off` (`eval.pos_label=0`).
- Backend: H2O AutoML (7,200s budget, GBM/XGB/StackedEnsemble eligible) via `make automl-h2o`.

## Summary Results (Test)
- ROC AUC: 0.7002
- Average Precision: 0.3841
- Threshold (Youden J on validation): 0.4722
- Charged Off (class 0) at threshold:
  - Precision: 0.8621
  - Recall: 0.6589
  - F1: 0.7469
  - Accuracy: 0.6520

See `results/metrics.json` and `results/confusion.json` for full details.

## Leaderboard (Held-out Test)
- Top models are GBM variants; best test AUC ≈ 0.7002 (`GBM_2_AutoML_1_20250924_13426`).
- Files:
  - `results/h2o_leaderboard.csv` (train leaderboard)
  - `results/h2o_leaderboard_test.csv` (test leaderboard)
  - `results/h2o_leaderboard.json` and `results/h2o_leaderboard_extra.csv` (extra diagnostics)

## Figures
- ROC curve: `figures/roc_curve.png`
- PR curve: `figures/pr_curve.png`
- Leaderboard metrics: `figures/h2o_leaderboard_auc.png`, `figures/h2o_leaderboard_logloss.png`, `figures/h2o_leaderboard_rmse.png`

## Reprovenance
- Resolved configuration: `results/config_resolved.yaml`
- Data manifest: `results/data_manifest.json`
- Feature manifest: `results/features.json`
- Environment lock: `results/requirements.freeze.txt`

## Artifacts
- Core results: `results/metrics.json`, `results/confusion.json`, `results/threshold_metrics.csv`
- H2O leaderboards: `results/h2o_leaderboard*.{csv,json}`
- Lightweight logs: `logs/` (warnings, httpd summary)

Large curve point CSVs (PR/ROC) are intentionally omitted from this folder to keep the repo lean. They can be regenerated from the saved model or referenced in the original run directory under `local_runs/run_20250924_013213/` if needed.

## How This Maps to the Proposal
The original proposal (moved from `docs/experiments/proposed/`) is saved as `PROPOSAL.md` alongside this report and is implemented with the time-based split (`split.method: time`) and a 7,200s AutoML budget. All references in this report are self-contained and do not depend on `local_runs/` paths.
