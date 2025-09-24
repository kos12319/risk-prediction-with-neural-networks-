# H2O AutoML Full-Dataset Run

## Context
- Dataset: `data/raw/full/thesis_data_full.csv` (LendingClub accepted loans 2007-2018). Ensure the archive is synced via `git lfs pull` and extracted before launching the run.
- Split policy: chronological by `issue_d` with validation carved from the training period only; test set uses the most recent vintages. Oversampling remains confined to the training subset.
- Positive class: `eval.pos_label = 0` (Charged Off). Threshold selection follows the Youden J strategy on the validation slice, then applies the fixed threshold to test metrics.
- Backend: H2O AutoML via `make automl-h2o`, with all artifacts written to `local_runs/` and tracking routed to W&B (online mode).

## Goals
- Train an AutoML ensemble on the full LendingClub dataset to establish a non-neural baseline and surface high-performing tree-based/blended models.
- Capture a leaderboard with extended diagnostics (test-set scores, curve data) and SHAP-style summaries for interpretability.
- Produce artifacts suitable for later comparison against PyTorch baselines, including W&B logs, H2O checkpoints, and evaluation reports.

## Next Run Notes
- The prior attempt hit the AutoML runtime cap before deep learning grids converged and cancelled several GBM sweeps. Bump `automl.max_runtime_secs` to at least 14,400 (4h) or relax it to `0` when compute allows so late-stage models finish.
- Install the `xgboost` Python wheel (or add the optional system dependency) ahead of launch so H2O can include XGBoost models rather than skipping them at startup.

## Configuration
- Config path: `configs/h2o/full_dataset.yaml`
- Key settings:
  - Time-based split (`split.method: time`) to respect temporal leakage invariants.
  - AutoML runtime budget: 7,200 seconds (~2 hours) to let all model classes converge.
  - JVM memory cap: `12G`, leaving headroom on 16 GB machines.
  - Thread limit: `nthreads: 8` to balance throughput vs. contention; adjust downward if the host has fewer physical cores.
  - Checkpoints and logs write under the run directory (`checkpoints/`, `h2o_logs/`).
  - W&B tracking is enabled (`tracking.backend: wandb`, `tracking.wandb.enabled: true`) with project `loan-risk-h2o` and descriptive run names including the winning algorithm ID.

```yaml
# configs/h2o/full_dataset.yaml
extends: ../default

data:
  csv_path: data/raw/full/thesis_data_full.csv

split:
  method: time
  time_col: issue_d
  test_size: 0.2
  random_state: 42
  stratify: false

model:
  backend: h2o

automl:
  max_runtime_secs: 7200
  max_models: null
  balance_classes: false
  seed: 42
  stopping_metric: AUC
  sort_metric: AUCPR
  include_algos: []
  exclude_algos: []
  nthreads: 8
  max_mem_size: 12G
  export_checkpoints_dir: checkpoints
  log_dir: h2o_logs
  leaderboard_extra_columns: ALL
  leaderboard_make_test: true
  leaderboard_curve_top_n: 10
  explanation_plots:
    model_correlation: true
    varimp_heatmap: true
    shap_summary: true

tracking:
  backend: wandb
  wandb:
    enabled: true
    mode: online
    project: loan-risk-h2o
    run_name_template: "{dataset}|{split}|{pos}|h2o[{leader_algo}]|{leader_id}|auc{auc:.3f}"
    tag_templates:
      - "backend:{backend}"
      - "leader_algo:{leader_algo_raw}"
      - "dataset:{dataset}"
```

## Commands
Run from the project root after exporting `WANDB_API_KEY` and `WANDB_ENTITY`:

```bash
make wandb-login
make automl-h2o AUTOML_CONFIG=configs/h2o/full_dataset.yaml NOTES="Full LendingClub AutoML sweep"
```

Add `PULL=true` to the second command if you want W&B artifacts downloaded into the run directory once training completes.

## Pre-Flight Checklist
- Environment:
  - `make venv` (or ensure `.venv` is up to date) and activate if you need manual CLI access.
  - Confirm the environment variables mentioned above are set before logging into W&B.
- Data integrity: verify `data/raw/full/thesis_data_full.csv` exists and matches expected row counts (no truncated downloads).
- Disk space: reserve >15 GB for AutoML artifacts, logs, and checkpoints inside `local_runs/`.
- Optional tuning: lower `automl.nthreads` or `max_mem_size` if the host has fewer than 8 cores or <16 GB RAM.

## Expected Outputs
- `local_runs/run_YYYYMMDD_HHMMSS/` containing:
  - `metrics.json`, `confusion.json`, `reports/` (ROC/PR plots, threshold report).
  - `h2o_leaderboard.csv/json` plus `h2o_leaderboard_test.csv` with held-out scores.
  - H2O checkpoints zipped under `checkpoints/` and raw logs in `h2o_logs/`.
  - W&B run metadata (online) tied to project `loan-risk-h2o`.
