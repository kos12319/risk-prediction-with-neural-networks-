# Thesis Iteration 1 — H2O Feature Regime Sweep

This suite runs H2O AutoML across four feature regimes for multiple dataset sizes:
- agnostic — provider-agnostic baseline
- selected — subset from feature selection (MI or L1)
- aware — includes provider pricing/scoring features
- selected_plus_providers — selected subset plus provider features

Makefile-first commands (examples):

1) Selection on 1k (generate `selected_features`):
- `make select CONFIG=configs/default.yaml METHOD=mi`
- `make select CONFIG=configs/default.yaml METHOD=l1`

2) Run H2O on 1k (time‑budgeted):
- `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/agnostic_time.yaml`
- `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/selected_time.yaml`
- `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/aware_time.yaml`
- `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/selected_plus_providers_time.yaml`

Helper scripts (from repo root):
- 1k: `./docs/experiments/suites/thesis_iter1/run_1k_sweep.sh [-n "notes"] [--pull]` (60s/run)
- 10k: `./docs/experiments/suites/thesis_iter1/run_10k_sweep.sh [-n "notes"] [--pull]` (300s/run)
- 100k: `./docs/experiments/suites/thesis_iter1/run_100k_sweep.sh [-n "notes"] [--pull]` (900s/run)
- full: `./docs/experiments/suites/thesis_iter1/run_full_sweep.sh [-n "notes"] [--pull]` (5400s/run)

3) Scale up (10k, 100k, full) using the `_time.yaml` configs under `h2o/<size>/` with budgets: 10k=300s, 100k=900s, full=5400s.

Notes:
- Ensure dataset paths exist locally (see README about LFS and samples). The 100k CSV may need to be unzipped to `data/raw/samples/thesis_data_sample_100k.csv`.
- For stability: keep time-based split and `eval.pos_label: 0` across all runs.
