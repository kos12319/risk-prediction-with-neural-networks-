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

2) Run H2O on 1k:
- `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/agnostic.yaml`
- `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/selected.yaml`
- `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/aware.yaml`
- `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/selected_plus_providers.yaml`

3) Scale up (10k, 100k, full) using analogous configs under `h2o/<size>/`.

Notes:
- Ensure dataset paths exist locally (see README about LFS and samples). The 100k CSV may need to be unzipped to `data/raw/samples/thesis_data_sample_100k.csv`.
- For stability: keep time-based split and `eval.pos_label: 0` across all runs.

