# Thesis Proposal — Iteration 1: Feature Set Optimization with H2O AutoML

## Objective
- Select a compact, high-value feature subset using the documented selection pipeline, then evaluate H2O AutoML performance across four feature regimes:
  1) Provider-agnostic (baseline)
  2) Selected subset (from MI/L1 selection)
  3) Provider-aware (includes pricing/scoring fields)
  4) Selected subset + provider features
- Validate on a 1k sample first; then scale to 10k → 100k → full dataset.

## Hypothesis
- A carefully selected subset achieves near-baseline performance while reducing complexity.
- Provider-aware signals (e.g., `int_rate`, `grade`, `sub_grade`, `installment`) materially improve performance; combining them with a compact subset offers the best accuracy/latency tradeoff.

## Methods
- Selection methods per docs: Mutual Information (filter) and L1-regularized logistic (embedded), with coverage-based stopping against the full-features AUC.
- Evaluation invariants upheld end-to-end:
  - Time-based split by `issue_d` (older → train, newer → test)
  - Hold-out validation carved from the training period only
  - Oversampling on training subset only
  - Threshold chosen on validation via configured strategy and applied to test
  - `eval.pos_label = 0` (Charged Off) respected in curves/metrics
  - Seeds set for Python/NumPy/Sklearn/H2O where supported

## Experiments
- Suite directory: `docs/experiments/suites/thesis_iter1/`
- Backends: H2O AutoML only (PyTorch unchanged in this iteration)
- Dataset ladder: 1k (smoke confirm) → 10k → 100k → full
- Per dataset, run four configs:
  - `h2o/{size}/agnostic.yaml` — provider-agnostic baseline
  - `h2o/{size}/selected.yaml` — selection-derived subset
  - `h2o/{size}/aware.yaml` — includes provider pricing/scoring
  - `h2o/{size}/selected_plus_providers.yaml` — selected subset + provider features

### 1k confirmation (this PR/run)
- Selection on 1k via `make select` (MI and/or L1); record `selected_features`.
- H2O AutoML runs with moderate budget (e.g., 600s) to ensure convergence on small data.
- Commands (Makefile-first):
  - `make select CONFIG=configs/default.yaml METHOD=mi`
  - `make select CONFIG=configs/default.yaml METHOD=l1`
  - `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/agnostic.yaml`
  - `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/selected.yaml`
  - `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/aware.yaml`
  - `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/selected_plus_providers.yaml`

### Scale-up plan
- 10k: increase AutoML budget (e.g., 1800s) and reuse the same four configs per size.
- 100k: point `csv_path` to `data/raw/samples/thesis_data_sample_100k.csv` (ensure the file exists locally; see README/LFS notes) and extend budget (e.g., 3600s+).
- Full: use `configs/h2o/full_dataset.yaml` as base; keep `leaderboard_extra_columns`, test leaderboard, and SHAP/varimp plots enabled.

## Success Criteria
- On 1k and 10k, selected subset reaches ≥98–99% of agnostic AUC/PR with fewer inputs.
- Provider-aware improves both AUC and PR; selected+providers gives the best Pareto (accuracy vs latency).
- Consistent thresholding and pos_label handling across all runs.

## Risks & Mitigations
- Small-sample variance on 1k: verify with 10k before drawing conclusions.
- H2O algos availability varies by platform: if XGBoost is unavailable, document and proceed; GBM/DRF/StackedEnsemble still provide strong baselines.
- Data readiness (100k/full): ensure archives are pulled/unzipped via LFS before launching.

## Artefacts & Reporting
- Each run writes a `local_runs/<id>/` folder with metrics, curves, config snapshot, and H2O leaderboards.
- Suite-level comparison tables/plots can be generated post hoc from `h2o_leaderboard_test.csv` and `metrics.json` across runs.

