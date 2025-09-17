# AGENTS — Project Onboarding and Working Guide

This file orients any agent landing in this repo with zero context.

## TL;DR (What this is)
- MSc thesis project: credit risk prediction on originated loans using a PyTorch MLP.
- Time‑based evaluation by loan issue date (`issue_d`) to avoid leakage.
- Everything is config‑driven; do not hardcode paths or parameters.

## Project Structure
- `src/` Python package
  - `data/` loading, target mapping, time/random split, simple feature engineering
  - `features/` preprocessing (impute, scale, one‑hot)
  - `models/` PyTorch MLP builder
  - `training/` orchestration, metrics, figures, run artifacts
  - `eval/` metrics and plotting utils
  - `cli/` runnable entry points (train, dryrun, select, column dict)
- `configs/` YAML configs (provider‑agnostic/aware)
- `data/`
  - `raw/archives/` original downloads and compressed files (Git LFS)
    - `kaggle_accepted_2007_to_2018Q4.csv.gz`, `kaggle_rejected_2007_to_2018Q4.csv.gz`
    - `full data set.zip` (original full dataset archive)
  - `raw/full/` canonical unzipped datasets (gitignored)
    - `thesis_data_full.csv` (accepted loans; 2007‑06 → 2018‑12)
    - `kaggle_accepted_2007_to_2018Q4.csv`, `kaggle_rejected_2007_to_2018Q4.csv`
  - `raw/samples/` small CSVs for quick runs (tracked)
    - `thesis_data_sample_100.csv`, `thesis_data_sample_1k.csv`, `thesis_data_sample_10k.csv`
    - `thesis_data_sample_100k.csv` (gitignored) and `thesis_data_sample_100k.zip` (LFS)
- `models/`, `reports/` artifacts; `docs/` ADRs/pain points/notes; `notebooks/` EDA

## Setup and Common Tasks
- Create venv and install deps
  - `python -m venv .venv && . .venv/bin/activate`
  - `pip install -r requirements.txt`
- Fetch large files (one‑time)
  - `git lfs install && git lfs pull`
  - Archives live in `data/raw/archives/` and are LFS‑tracked; unzipped full datasets under `data/raw/full/` are ignored.
- Train (config‑driven)
  - `python -m src.cli.train --config configs/default.yaml`
  - Edit `data.csv_path` in the config to point at a CSV (e.g., `data/raw/samples/thesis_data_sample_10k.csv` or `data/raw/full/thesis_data_full.csv`).
- Dry run (no artifacts written; prints JSON summary)
  - `python -m src.cli.dryrun --config configs/default.yaml`
- Feature selection (MI or L1)
  - `python -m src.cli.select --config configs/default.yaml --method mi`
- Column dictionary (types, missingness, categories)
  - `python -m src.cli.gen_column_dict --config configs/default.yaml`
- Makefile shortcuts
  - `make venv | train | select | dict | dryrun` with `CONFIG=...` and `METHOD=mi|l1`

## Evaluation Invariants (don’t break)
- Use time‑based split by `issue_d` for test; older → train, newer → test.
- Hold out validation from the training period; only oversample the training subset.
- Choose threshold based on configured strategy (`fixed|youden_j|f1`) on validation; report test metrics at that fixed threshold.
- Respect `eval.pos_label` (default: 0 = Charged Off). Curves/metrics should reflect the configured positive class.
- Seed randomness for reproducibility (Python, NumPy, PyTorch, DataLoader workers).

## Run Artifacts (thesis reproducibility)
Every training run writes `reports/runs/run_YYYYMMDD_HHMMSS/` containing:
- `README.md`: summary with config, backend, pos label, threshold strategy/choice, ROC AUC, AP, confusion stats, dataset info.
- `metrics.json`, `confusion.json`: metrics and confusion at the chosen threshold.
- `figures/`: `learning_curves.png`, `roc_curve.png`, `pr_curve.png`.
- `config_resolved.yaml`: final config (after `extends`/overrides).
- `features.json`: numerical/categorical lists and final feature inputs.
- `history.csv`: epochs with `loss`/`val_loss`.
- `roc_points.csv`, `pr_points.csv`: threshold sweeps.
- `data_manifest.json`: dataset path, sha256/size/mtime, class counts, date ranges.
- Model artifact (e.g., `loan_default_model.pt`). Latest copies also live under `reports/` and `reports/figures/`.

## Data Handling & LFS
- Do not commit large uncompressed data outside `data/raw/archives/` (LFS) or `data/raw/full/` (ignored).
- Archives are tracked by LFS (`*.zip` and `data/raw/archives/**`).
- Samples are tracked, except `thesis_data_sample_100k.csv` which is gitignored; its zip lives in samples and is LFS‑tracked.
- If archives are LFS pointers locally, run `git lfs pull`; then unzip into `data/raw/full/` or `data/raw/samples/` as appropriate.

## Coding Style & Conventions
- Python 3.10+; type hints required for public functions.
- Naming: `snake_case` (functions/variables), `PascalCase` (classes), `UPPER_SNAKE` (constants).
- Keep modules cohesive and small; prefer pure functions over side effects.
- Avoid data leakage by design: time‑split, train‑only oversampling, and drop post‑origination features (`data.drop_leakage`).

## Testing
- Use `pytest`; place tests under `tests/` as `test_*.py`.
- Suggested tests: preprocessing invariants, time‑split monotonicity, model I/O round‑trip, thresholding correctness, run artifacts schema.

## PR Hygiene
- Commits: concise, imperative subject; group related changes.
  - Examples: `feat(training): add focal loss option`, `fix(data): compute credit_history_length vs issue_d`.
- PRs: include what/why, config used, before/after ROC AUC (from run `metrics.json`), and figures under `reports/figures/`.

## Known TODOs / Watch‑outs
- Oversampling isolation: ensure validation is carved from train before oversampling; oversample train subset only.
- Determinism: seed Python/NumPy/Torch; use a seeded generator for Torch splits.
- Threshold selection: compute on validation; apply fixed threshold to test.
- Dense conversion: prefer `.toarray()` over `.todense()` where applicable.
- Feature name mapping after OHE: avoid brittle string splits; use encoder introspection.
- Selection CLI: ensure it resolves `extends` like training does.
- Headless plotting: use `MPLBACKEND=Agg`; set `XDG_CACHE_HOME=.cache` and `MPLCONFIGDIR=.mplcache` if needed. Limit BLAS threads if you see OMP errors.

## If You’re Lost
1) `make venv` then `make dryrun CONFIG=configs/default.yaml`.
2) If data missing, run `git lfs pull` and point `data.csv_path` to one of the samples under `data/raw/samples/`.
3) Read `docs/adr/` (time split rationale and PyTorch choice) and `docs/PAIN_POINTS.md`.
4) Ask for clarification before changing evaluation protocols or data handling.
