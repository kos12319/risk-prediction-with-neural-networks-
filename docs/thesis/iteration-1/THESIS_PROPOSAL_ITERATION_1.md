# Thesis Proposal — Iteration 1: Feature-Set Optimization (H2O AutoML)

## Objective
- Compare four feature regimes across data scales to quantify the lift from provider-aware signals and compact selection:
  1) Provider-agnostic (baseline)
  2) Selected subset (from MI/L1 selection)
  3) Provider-aware (adds `int_rate`, `grade/sub_grade`, `installment`)
  4) Selected subset + provider-aware
- Start at 1k for smoke confirmation; scale to 10k → 100k → full.

## Hypotheses
- Selected subset achieves ≥98–99% of baseline AUCPR on 1k/10k with fewer inputs.
- Provider-aware signals materially improve AUCPR on 10k/100k/full; selected+providers offers the best accuracy/latency trade-off.

## Methods
- Selection per docs: Mutual Information (filter) and L1-logistic (embedded) with coverage gating vs full-features AUC.
- Evaluation invariants (Makefile-first):
  - Time split by `issue_d` (older → train, newer → test)
  - Validation carved from train only; oversample train subset only
  - Threshold chosen on validation (e.g., Youden J) and fixed on test
  - Respect `eval.pos_label = 0` (Charged Off) in all metrics/curves
  - Seed Python/NumPy/H2O; headless plotting, thread limits via Makefile

## Experiments
- Canonical suite: `docs/experiments/suites/thesis_iter1/` (configs + reports)
- Thesis snapshot: `docs/thesis/iteration-1/` (narrative + key figures)
- Backend: H2O AutoML (PyTorch unchanged in this iteration)
- Sizes: 1k, 10k, 100k, full. Per size, run four configs:
  - `h2o/{size}/agnostic.yaml`
  - `h2o/{size}/selected.yaml`
  - `h2o/{size}/aware.yaml`
  - `h2o/{size}/selected_plus_providers.yaml`

### 1k confirmation
- Run selection on 1k via `make select` (MI and/or L1); capture `selected_features`.
- H2O AutoML with moderate budget (≈600s) to converge on small data.
- Commands (Makefile-first):
  - `make select CONFIG=configs/pytorch_default.yaml METHOD=mi`
  - `make select CONFIG=configs/pytorch_default.yaml METHOD=l1`
  - `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/agnostic.yaml`
  - `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/selected.yaml`
  - `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/aware.yaml`
  - `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/selected_plus_providers.yaml`

### Scale-up plan
- 10k: increase AutoML budget (≈1800s) and reuse the four configs.
- 100k: set `csv_path` to `data/raw/samples/thesis_data_sample_100k.csv` and extend budget (≥3600s).
- Full: use H2O defaults; keep leaderboard extras, PR/ROC curves, varimp heatmaps.

## Success Criteria
- Selected subset reaches ≥98–99% of agnostic AUC/PR on 1k/10k.
- Provider-aware and selected+providers improve AUCPR on 10k/100k/full.
- Consistent thresholding and `pos_label` handling across runs.

## Risks & Mitigations
- Small-sample variance at 1k → confirm on 10k before conclusions.
- Platform variance (e.g., missing XGBoost) → proceed with GBM/DRF/SE and document.
- Data readiness for 100k/full → pull LFS archives before runs.

## Artefacts & Reporting
- Each run writes `local_runs/<id>/` with metrics, curves, config snapshot, and H2O leaderboards.
- Suite-level plots in `docs/experiments/suites/thesis_iter1/reports/`; thesis copies in `docs/thesis/iteration-1/reports/`.
