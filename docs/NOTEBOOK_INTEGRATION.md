# Mapping main.ipynb to the Project

This document explains how the work done in `main.ipynb` has been incorporated into the structured, config‑driven project.

## Data and Target
- Uses the same binary target mapping: `loan_status` → {Fully Paid: 1, Charged Off: 0}.
- Reads a configurable subset of origination‑time features from CSV (see `configs/*.yaml`).
- Parses dates (`issue_d`, `earliest_cr_line`) for time‑aware splitting and feature engineering.

## Preprocessing
- ColumnTransformer mirrors the notebook:
  - Numeric: median imputation → StandardScaler.
  - Categorical: most‑frequent imputation → OneHotEncoder(`handle_unknown='ignore'`, sparse).
- Keeps matrices sparse through preprocessing; converts to dense before feeding Keras.

## Balancing
- Same RandomOverSampler approach, but applied only to the training split to avoid leakage (the notebook oversampled before splitting).

## Engineered Features
- `credit_history_length`: computed as months between `earliest_cr_line` and `issue_d` (fix from notebook’s use of “today”).
- `income_to_loan_ratio` = `annual_inc / loan_amnt`.
- `fico_avg` and `fico_spread` from `fico_range_low/high`.

## Model
- Keras MLP similar to the stronger notebook variant: 256 → 128 → 64 → 32 with BatchNorm and Dropout; sigmoid output.
- Loss: `binary_crossentropy` by default; optional focal loss is available via config.
- EarlyStopping on `val_loss`; learning curves plotted and saved.

## Evaluation
- Saves ROC AUC and classification report to `reports/metrics.json`.
- Saves learning curve figure to `reports/figures/learning_curves.png`.

## Feature Sets and Missingness
- Provider‑agnostic config (default): excludes lender pricing/scoring (`int_rate`, `grade`, `sub_grade`, `installment`) and `funded_amnt`.
- Provider‑aware config: includes those fields for comparison.
- Based on `missing_data_summary.csv`, drops `inq_last_12m` by default (~41% missing). Other suggested features show ≤~3.5% missingness and are kept.

## Splitting and Leakage Controls
- Time‑based split by `issue_d` (default), or random stratified split via config.
- Drops post‑origination fields (repayment/collection/hardship/settlement) to prevent leakage.

## Reproducibility and CLI
- Config‑driven runs: `python -m src.cli.train --config configs/default.yaml`.
- Configs support `extends` (e.g., `provider_agnostic.yaml` can inherit from `default.yaml`).
- Artifacts are saved under `models/` and `reports/`.

## Intentional Differences vs Notebook
- Oversampling moved after the split (fixes leakage).
- Time‑based split option added; recommended for credit origination modeling.
- `credit_history_length` computed relative to `issue_d`.
- Explicit leakage column dropping.
- Engineered features added as first‑class options in the loader.

