Credit Risk Prediction with Neural Networks and Feature Subset Selection

Overview
- Thesis goal: build a neural-network–based credit risk model and select a compact, high‑value feature subset.
- Organized from your working notebook into a configurable, reproducible pipeline.

Repository Layout
- data/
  - raw/
    - archives/ — original downloads and compressed files (tracked via Git LFS)
      - kaggle_accepted_2007_to_2018Q4.csv.gz, kaggle_rejected_2007_to_2018Q4.csv.gz
      - full data set.zip (original full‑dataset archive)
    - full/ — canonical unzipped datasets (ignored by git)
      - thesis_data_full.csv (accepted loans, 2007‑06 → 2018‑12)
      - kaggle_accepted_2007_to_2018Q4.csv, kaggle_rejected_2007_to_2018Q4.csv
    - samples/ — small CSVs for quick runs (tracked; the 100k CSV is ignored)
      - thesis_data_sample_100.csv, thesis_data_sample_1k.csv, thesis_data_sample_10k.csv
      - thesis_data_sample_100k.csv (gitignored) and thesis_data_sample_100k.zip (LFS‑tracked)
  - processed/: optional cached splits (ignored by git)
- models/: saved models (e.g., loan_default_model.pt)
- reports/
  - figures/: learning curves and plots
- configs/
  - default.yaml: data paths, features, split, and model settings
- src/
  - data/: loading, cleaning, splitting
  - features/: preprocessing and feature engineering
  - models/: neural network definition (PyTorch)
  - eval/: metrics and plots
  - training/: training orchestration
  - cli/: command-line entry points
- notebooks/: exploratory notebooks

Docs
- Agent Guide: see `AGENTS.md` for onboarding and working instructions.
- ADRs: see `docs/adr/` for architecture decision records.
- Pain Points: see `docs/PAIN_POINTS.md` for current issues and recommendations.
- Data dictionary: see `docs/data/COLUMN_DICTIONARY.md` for per-column types, leakage flags, and descriptions based on the sample CSV.
- Regenerate column dictionary:
   - Via Makefile: `make venv && . .venv/bin/activate && python -m src.cli.gen_column_dict --config configs/default.yaml` (or use `--csv` to override).

Dry Run (no artifacts stored)
- Use this to check an experiment end-to-end without persisting any files:
  - Via Makefile: `make dryrun CONFIG=configs/default.yaml`
  - Direct: `. .venv/bin/activate && python -m src.cli.dryrun --config configs/default.yaml`
- All model and report paths are redirected to a temporary directory that is deleted after the run. The CLI prints a JSON summary to stdout.

Quickstart
1) Ensure Python 3.12 is available (preferred for PyTorch wheels). Then create a venv and install deps:
   # via Makefile (recommended)
   make venv
   # or manually
   python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

2) Choose a config (project target = NN + feature subset selection):
   - Provider-agnostic (default): configs/default.yaml or configs/provider_agnostic.yaml
     Excludes lender pricing/scoring fields (int_rate, grade, sub_grade, installment) and funded_amnt.
   - Provider-aware: configs/provider_aware.yaml
     Includes int_rate, grade, sub_grade, installment.
   Update data.csv_path to your CSV (e.g., `data/raw/samples/thesis_data_sample_10k.csv` for quick runs, or your local full file under `data/raw/full/thesis_data_full.csv`). The 100k sample is available as `data/raw/samples/thesis_data_sample_100k.csv` (ignored) and `data/raw/samples/thesis_data_sample_100k.zip` (LFS).

3) Train the model:
   # via Makefile
   make train CONFIG=configs/default.yaml
   # or directly
   python -m src.cli.train --config configs/default.yaml

Artifacts
- Model: models/loan_default_model.pt
- Metrics: reports/metrics.json
- Learning curves: reports/figures/learning_curves.png
 - Per-run history: see `reports/runs/run_*/` (details in `docs/RUN_ARTIFACTS.md`), including `data_manifest.json` with dataset provenance and date ranges.

Feature Selection
- Mutual Information or L1-logistic selection with incremental AUC evaluation:
  - `python -m src.cli.select --config configs/default.yaml --method mi`
  - `python -m src.cli.select --config configs/default.yaml --method l1`
- Outputs under `reports/selection/<method>/`: ranked list (CSV/JSON), selected subset, and AUC curve plot.

Feature Subset Selection (scope)
- The project targets identifying a minimal subset of origination-time features with near‑maximal predictive power.
- Planned/typical approaches: filter (missingness/variance/MI), embedded (L1/logistic, tree importances), wrappers (RFECV or sequential selection) evaluated via time‑aware validation.
- Results should report performance with all features vs selected subset, and provider‑agnostic vs provider‑aware.

Notes
- Oversampling is applied only to the training split to avoid leakage.
- Engineered features: credit_history_length (months from earliest_cr_line to issue_d), income_to_loan_ratio,
  fico_avg and fico_spread are computed when inputs are present.
- Post-origination columns are dropped by default (configurable).

Notebook Integration
- See docs/NOTEBOOK_INTEGRATION.md for a clear mapping from your original main.ipynb to this project, including what was preserved and the intentional fixes.
