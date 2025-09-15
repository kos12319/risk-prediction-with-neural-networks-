Credit Risk Prediction with Neural Networks and Feature Subset Selection

Overview
- Thesis goal: build a neural-network–based credit risk model and select a compact, high‑value feature subset.
- Organized from your working notebook into a configurable, reproducible pipeline.

Repository Layout
- data/
  - raw/: place original CSV(s)
  - processed/: optional cached splits
- models/: saved models (e.g., loan_default_model.h5)
- reports/
  - figures/: learning curves and plots
- configs/
  - default.yaml: data paths, features, split, and model settings
- src/
  - data/: loading, cleaning, splitting
  - features/: preprocessing and feature engineering
  - models/: neural network definition
  - eval/: metrics and plots
  - training/: training orchestration
  - cli/: command-line entry point
- notebooks/: keep exploratory notebooks (optional)

Quickstart
1) Ensure Python 3.10+ and install dependencies:
   pip install -r requirements.txt

2) Choose a config (project target = NN + feature subset selection):
   - Provider-agnostic (default): configs/default.yaml or configs/provider_agnostic.yaml
     Excludes lender pricing/scoring fields (int_rate, grade, sub_grade, installment) and funded_amnt.
   - Provider-aware: configs/provider_aware.yaml
     Includes int_rate, grade, sub_grade, installment.
   Update data.csv_path to your CSV (e.g., ./first_10k_rows.csv or the full dataset path).

3) Train the model:
   python -m src.cli.train --config configs/default.yaml
   # or
   python -m src.cli.train --config configs/provider_aware.yaml

Artifacts
- Model: models/loan_default_model.h5
- Metrics: reports/metrics.json
- Learning curves: reports/figures/learning_curves.png

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
