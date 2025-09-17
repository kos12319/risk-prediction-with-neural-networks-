# Credit Risk Prediction with Neural Networks and Feature Subset Selection

## Overview
- Build a neural-network–based credit risk model and select a compact, high‑value feature subset.
- Configurable, reproducible pipeline extracted from notebooks.
- All local artifacts write to a single gitignored folder per run under `local_runs/`.

## Project Layout
- `data/`
  - `raw/`
    - `archives/` — original downloads and compressed files (tracked via Git LFS)
      - `kaggle_accepted_2007_to_2018Q4.csv.gz`, `kaggle_rejected_2007_to_2018Q4.csv.gz`
      - `full data set.zip` (original full‑dataset archive)
    - `full/` — canonical unzipped datasets (ignored by git)
      - `thesis_data_full.csv` (accepted loans, 2007‑06 → 2018‑12)
      - `kaggle_accepted_2007_to_2018Q4.csv`, `kaggle_rejected_2007_to_2018Q4.csv`
    - `samples/` — small CSVs for quick runs (tracked; the 100k CSV is ignored)
      - `thesis_data_sample_100.csv`, `thesis_data_sample_1k.csv`, `thesis_data_sample_10k.csv`
      - `thesis_data_sample_100k.csv` (gitignored) and `thesis_data_sample_100k.zip` (LFS‑tracked)
  - `processed/` — optional cached splits (ignored by git)
- `local_runs/` (gitignored) — per‑run folders with all artifacts
- `configs/` — YAML configs; `default.yaml` is the main one
- `src/`
  - `data/` — loading, cleaning, splitting
  - `features/` — preprocessing and feature engineering
  - `models/` — neural network (PyTorch)
  - `eval/` — metrics and plots
  - `training/` — training orchestration
  - `cli/` — command‑line entry points
- `notebooks/` — exploratory notebooks

## Local Artifacts
- All new runs save to `local_runs/run_YYYYMMDD_HHMMSS/` (gitignored).
- Legacy `reports/` and `models/` are deprecated and ignored.
- Each run folder contains model, metrics, figures, config snapshot, provenance, and optionally a `wandb/` subfolder with downloaded W&B data.

## Documentation
- Agent Guide: `AGENTS.md`
- ADRs: `docs/adr/`
- Pain Points: `docs/PAIN_POINTS.md`
- Data dictionary: `docs/data/COLUMN_DICTIONARY.md`
- Regenerate column dictionary:
  ```bash
  make venv
  . .venv/bin/activate
  python -m src.cli.gen_column_dict --config configs/default.yaml  # or use --csv
  ```

## Dry Run
- End‑to‑end check without persisting files:
  ```bash
  make dryrun CONFIG=configs/default.yaml
  # or
  . .venv/bin/activate
  python -m src.cli.dryrun --config configs/default.yaml
  ```
- Artifacts are written to a temporary directory and deleted after completion. A JSON summary is printed to stdout.

## Quick Start
1) Create venv and install deps (Python 3.12 preferred):
   ```bash
   make venv
   # or
   python3.12 -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   ```
2) Choose a config and set dataset path:
   - Provider‑agnostic (default): `configs/default.yaml` (excludes int_rate/grade/sub_grade/installment and funded_amnt)
   - Provider‑aware: `configs/provider_aware.yaml` (includes pricing/scoring fields)
   - Set `data.csv_path` to a CSV (e.g., `data/raw/samples/thesis_data_sample_10k.csv` or `data/raw/full/thesis_data_full.csv`)
3) Login to W&B from env (optional, needed for downloads):
   ```bash
   export WANDB_API_KEY=...    # required to pull/download
   export WANDB_ENTITY=your_entity
   # optional: export WANDB_PROJECT=loan-risk-mlp
   make wandb-login
   ```
4) Train the model:
   ```bash
   make train CONFIG=configs/default.yaml             # just train
   make train CONFIG=configs/default.yaml PULL=true   # train and download W&B files
   # On Linux/WSL or constrained envs, use CPU-only helper:
   make cpu-train CONFIG=configs/default.yaml         # CPU with minimal threads
   ```
5) Download any W&B run later (to a local folder):
   ```bash
   make pull-run RUN=entity/project/run_id            # default: local_runs/<run_id>/wandb/
   # or specify explicit destination
   make pull-run RUN=entity/project/run_id TARGET=/path/to/folder
   ```

## Artifacts
- Location: `local_runs/run_YYYYMMDD_HHMMSS/`
- Files:
  - Model: `loan_default_model.pt`
  - Metrics: `metrics.json` (ROC AUC, AP, threshold, classification report)
  - Confusion: `confusion.json` (TP/FP/TN/FN, precision/recall/specificity)
  - Curves: `figures/learning_curves.png`, `figures/roc_curve.png`, `figures/pr_curve.png`
  - Sweeps: `roc_points.csv`, `pr_points.csv`
  - Provenance: `config_resolved.yaml`, `features.json`, `data_manifest.json`, `requirements.freeze.txt`, `training.log`
  - W&B: `wandb.json` with `{id, path, url}`; optional `wandb/` with downloaded files/artifacts (when `PULL=true` or via `pull-run`)

## Feature Selection
- Mutual Information or L1‑logistic selection with incremental AUC evaluation:
  ```bash
  python -m src.cli.select --config configs/default.yaml --method mi
  python -m src.cli.select --config configs/default.yaml --method l1
  ```
- Outputs under `reports/selection/<method>/`: ranked list (CSV/JSON), selected subset, and AUC curve plot. See `docs/FEATURE_SELECTION.md`.

## Experiment Tracking (W&B)
- Enable in config: `tracking.backend: wandb`; `tracking.wandb.enabled: true`.
- Useful options (see `configs/default.yaml`):
  - `run_name` or `run_name_template` — placeholders: `{dataset},{split},{pos},{layers},{nf},{auc},{sha},{run_id}`
  - `group` or `group_template` — default: `{dataset}|{split}|{pos}`
  - `job_type`/`job_type_template`, `tags`/`tag_templates`, `ignore_globs`, `log_artifacts`
- Login via env: set `WANDB_API_KEY` and `WANDB_ENTITY`, then `make wandb-login` or just train (trainer auto‑logins if key is present). Optional `WANDB_PROJECT` overrides config.
- Download W&B data to local folder:
  - After training: `make train CONFIG=configs/default.yaml PULL=true` → `local_runs/<run_id>/wandb/`
  - Any time: `make pull-run RUN=entity/project/run_id [TARGET=dir]`
- Logged in W&B: per‑epoch loss/val_loss/val_auc/lr/time, final metrics (incl. confusion), env+git metadata, requirements snapshot, figures, interactive confusion matrix panel; key files and model are logged as artifacts.

## Environment Variables
- `WANDB_API_KEY` — required for W&B API login and downloads
- `WANDB_ENTITY` — your W&B user or org (used if not in config)
- `WANDB_PROJECT` — optional, overrides config project for new runs and downloads
- `FORCE_CPU=1` — force CPU training; Makefile `cpu-train` sets this automatically
- Thread controls (set by `cpu-train`): `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1`

## CLI Reference
- Train: `python -m src.cli.train --config CONFIG [--notes TEXT] [--pull] [--cpu]`
- W&B login: `python -m src.cli.wandb_login`
- Pull W&B run: `python -m src.cli.wandb_pull --run ENTITY/PROJECT/RUN_ID [--target DIR] [--config CONFIG]`

## Makefile Targets
- `make train CONFIG=... [PULL=true] [NOTES=...]`
- `make cpu-train CONFIG=... [PULL=true] [NOTES=...]`
- `make wandb-login`
- `make pull-run RUN=entity/project/run_id [TARGET=dir]`
- `make clean-artifacts` — removes `reports/`, `models/`, and `local_runs/`

## Feature Subset Selection (Scope)
- Goal: identify a minimal subset of origination‑time features with near‑maximal predictive power.
- Approaches: filter (missingness/variance/MI), embedded (L1/logistic, tree importances), wrappers (RFECV/sequential) with time‑aware validation.
- Report performance for all features vs selected subset, and provider‑agnostic vs provider‑aware.

## Notes
- Oversampling applies only to the training split to avoid leakage.
- Engineered features: `credit_history_length` (months from `earliest_cr_line` to `issue_d`), `income_to_loan_ratio`, `fico_avg`, `fico_spread`.
- Post‑origination columns are dropped by default (configurable).

## Notebook Integration
- See `docs/NOTEBOOK_INTEGRATION.md` for mapping from the original notebook to this project, including preserved parts and fixes.
