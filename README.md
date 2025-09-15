Credit Risk Prediction Project

Overview
- Predicts credit default risk using a neural network trained on LendingClub-style origination data.
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

2) Update configs/default.yaml:
   - Set data.csv_path to your CSV (e.g., ./first_10k_rows.csv or the full accepted dataset path)
   - Adjust feature list if needed

3) Train the model:
   python -m src.cli.train --config configs/default.yaml

Artifacts
- Model: models/loan_default_model.h5
- Metrics: reports/metrics.json
- Learning curves: reports/figures/learning_curves.png

Notes
- Oversampling is applied only to the training split to avoid leakage.
- credit_history_length is computed relative to issue_d (months).
- Post-origination columns are dropped by default (configurable).
