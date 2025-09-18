# Credit Risk Prediction with Neural Networks and Feature Subset Selection

## Overview
- Build a neural-network–based credit risk model and select a compact, high‑value feature subset.
- Configurable, reproducible pipeline extracted from notebooks.
- All local artifacts write to a single gitignored folder per run under `local_runs/`.

## Dataset (LendingClub)
- This project uses the LendingClub consumer installment loans dataset (2007–2018 vintages).
- Two public files exist: “accepted” loans (funded applications with final statuses) and “rejected” applications (declined, limited covariates).
- Labels are derived from funding outcomes (e.g., Charged Off vs Fully Paid). For recent vintages, be mindful of right‑censoring when interpreting “non‑defaults”.
- Modeling at origination strictly excludes post‑event fields (payments, recoveries, last_* dates, hardship/settlement) to prevent leakage.
- Default positive class convention in configs is `eval.pos_label: 0` (Charged Off); curves/metrics reflect this unless changed.

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
- ADRs: `docs/ADRs/`
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
   Notes:
   - CPU-only environments: if Torch wheel resolution fails or attempts CUDA, install a CPU wheel explicitly (e.g., pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu) before syncing other deps.
   - Apple Silicon: set VECLIB/OMP env already handled; use `make cpu-train` if you hit BLAS thread errors.
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
5) Download runs from W&B (to a separate history folder):
   ```bash
   # Pull a specific run (requires a run_id); resolves entity/project from config/env
   make pull-run RUN=entity/project/run_id            # downloads into wandb-history/<run_id>/
   make pull-run RUN=project/run_id                   # uses WANDB_ENTITY from env/config
   make pull-run RUN=run_id                           # uses WANDB_ENTITY and WANDB_PROJECT

   # Pull all runs for the configured project (skips existing folders by default)
   make pull-all                                      # downloads into wandb-history/<run_id>/
   make pull-all FORCE=1                              # force re-download/overwrite
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
- How to run (two options):
  - Makefile: `make select CONFIG=configs/default.yaml METHOD=mi` (or `METHOD=l1`)
  - Direct:
    ```bash
    python -m src.cli.select --config configs/default.yaml --method mi
    python -m src.cli.select --config configs/default.yaml --method l1
    ```
- Optional flags (direct invocation): `--target_coverage 0.99 --missingness_threshold 0.5 --max_features 50 --outdir reports/selection`
- Outputs under `reports/selection/<method>/`:
  - `*_results.json` — selected_features, full_AUC, incremental steps
  - `*_ranking.csv` — full ranking with scores
  - `*_auc_curve.png` — AUC vs number of features
- Apply the subset:
  1) Open `reports/selection/<method>/*_results.json`
  2) Copy `selected_features` into `data.features` in your YAML config (or create a new config variant)
  3) Train with that config and compare to the full set
- Details and method rationale: see `docs/FEATURE_SELECTION.md`.

## Experiment Tracking (W&B)
- Enable in config: `tracking.backend: wandb`; `tracking.wandb.enabled: true`.
- Useful options (see `configs/default.yaml`):
  - `run_name` or `run_name_template` — placeholders: `{dataset},{split},{pos},{layers},{nf},{auc},{sha},{run_id}`
  - `group` or `group_template` — default: `{dataset}|{split}|{pos}`
  - `job_type`/`job_type_template`, `tags`/`tag_templates`, `ignore_globs`, `log_artifacts`
- Login via env: set `WANDB_API_KEY` and `WANDB_ENTITY`, then `make wandb-login` or just train (trainer auto‑logins if key is present). Optional `WANDB_PROJECT` overrides config.
- Download W&B data to local folder:
  - After training: `make train CONFIG=configs/default.yaml PULL=true` → `local_runs/<run_id>/wandb/`
  - Any time: `make pull-run RUN=entity/project/run_id` → `wandb-history/<run_id>/`
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
- `make pull-run RUN=entity/project/run_id` — saves to `wandb-history/<run_id>/`
- `make pull-all [FORCE=1]` — saves all to `wandb-history/<run_id>/`
- `make clean-local-runs` — removes `local_runs/` only
- `make clean-wandb-local` — removes `./wandb` (local SDK logs/cache)
- `make clean-local-history` — removes `./wandb-history` (downloaded run histories)
- `make clean-all-local` — removes `local_runs/`, `./wandb`, and `./wandb-history`
- `make clean-cloud-history FORCE=1` — deletes all runs (and logged artifacts) from the configured W&B project

## Dependency Management
- This repo uses pip-tools with a two-file setup:
  - `requirements.in` — human-edited top-level deps (loose pins allowed)
  - `requirements.txt` — compiled, fully pinned lockfile
- Typical workflow:
  - Edit `requirements.in`
  - Install tools: `make deps-tools`
  - Compile lock: `make deps-compile` (updates `requirements.txt`)
  - Sync venv: `make deps-sync` (installs exactly the pinned set)
  - Alternatively, install directly: `pip install -r requirements.txt`
 - Contributors: prefer editing `requirements.in` and regenerating the lock (avoid hand-editing `requirements.txt`).


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
