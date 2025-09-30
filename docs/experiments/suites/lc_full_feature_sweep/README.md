Experiment suite: LendingClub full dataset — H2O AutoML feature sweeps

This suite recreates the H2O portion of the original LC full-feature sweep with just
four long-running variations. Each config extends the default H2O setup, keeps
winsorisation enabled, and caps AutoML to a 10 hour budget (36,000 seconds).

Included runs (configs live under `docs/experiments/suites/lc_full_feature_sweep/h2o/`):
- `provider_agnostic_all.yaml` — full provider-agnostic feature deck (baseline)
- `provider_agnostic_l1.yaml` — 12-feature L1 subset without lender pricing fields
- `provider_aware_l1.yaml` — L1 core subset plus `int_rate`, `grade`, `sub_grade`, `installment`
- `provider_aware_l1_cv.yaml` — same feature set as above with 5-fold expanding temporal CV

Use the helper script to queue the suite (it only orchestrates, do not run yet):
- `docs/experiments/suites/lc_full_feature_sweep/h2o/run_h2o_suite.sh`
  - Supports optional `--notes`, `--pull`, and `--resume` flags
  - Calls `make automl-h2o AUTOML_CONFIG=…` for each config sequentially
  - Logs outcomes to `docs/experiments/suites/lc_full_feature_sweep/h2o/run_h2o_suite.log`

Key settings baked into every config:
- Dataset: `data/raw/full/thesis_data_full.csv` (time split on `issue_d`)
- Winsorisation targets core numeric columns (`dti`, `loan_amnt`, `revol_bal`, etc.)
- AutoML runtime cap: 36,000 seconds (10 hours); class balancing left enabled
- Resource guardrails: AutoML pinned to 8 threads and a 12 GB JVM heap (`max_mem_size: 12G`)
- Ensembles skipped to conserve RAM (`exclude_algos: ['StackedEnsemble']`)
- Positive class, thresholding, and leakage guards inherit from the shared defaults
- CV variant uses 5 expanding folds with `train_full_after: true` after cross-validation

PyTorch companions from the legacy sweep were intentionally dropped; only the
H2O runs need to be revisited at this stage.
