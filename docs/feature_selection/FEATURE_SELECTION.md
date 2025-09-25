# Feature Selection Procedure

This document explains how the project selects a compact, high‑value subset of features for credit‑risk prediction, consistent with the evaluation rules used in training.

## Why We Do It
- Reduce dimensionality and training cost.
- Improve generalization by removing weak/redundant inputs.
- Produce interpretable subsets for reporting in the thesis.

## Evaluation Invariants (kept here, too)
- Time‑based split by `issue_d`: older → train, newer → test.
- Preprocess on the training data only; apply to test.
- Seed randomness for reproducibility (Python/NumPy/Sklearn).
- No experiment logging here; selection writes local artifacts only.

## Preprocessing Pipeline (same as training)
1) Numerical: impute missing, scale.
2) Categorical: one‑hot encode.
3) Fit the transformer on the training split; transform train/test.

Internally, many original columns expand to multiple encoded columns (one per category). Selection aggregates scores back to the original feature name to avoid bias toward high‑cardinality categoricals.

## Methods

### Mutual Information (MI)
- Intuition: score each feature by how informative it is about the target on its own (captures non‑linear signals without a model).
- Computation:
  - Compute MI per encoded column with `mutual_info_classif`.
  - Aggregate MI scores back to each original feature by summing across its encoded columns.
  - Rank features by aggregated MI.
- Pros: fast, model‑agnostic, catches non‑monotonic relations.
- Cons: univariate (ignores interactions); can favor features with many categories (mitigated by grouping).

### L1‑Regularized Logistic ("L1")
- Intuition: train a simple logistic model that prefers using fewer inputs; many coefficients become zero.
- Computation:
  - Fit logistic regression with `penalty='l1'`, solver `saga`, balanced classes.
  - Take absolute coefficients per encoded column; aggregate back to the original feature by summing magnitudes.
  - Rank features by aggregated |coef|.
- Pros: model‑based, handles correlated features by selecting a sparse set.
- Cons: the sparsity strength (`C`) affects which features are kept; can be unstable among highly correlated features.

## Subset Construction and Stopping Rule
1) Compute a reference ROC AUC using all filtered features (post missingness filter).
2) Add features in ranked order and, at each step, re‑fit a fast logistic baseline; compute test AUC.
3) Stop when subset AUC ≥ `target_coverage × full_AUC` (e.g., 0.99), or when `max_features` is reached.

Notes:
- This is a pragmatic wrapper that uses the test split to decide the subset size for speed. For stricter protocol, switch to a validation split carved from the training period to choose K, then report final test metrics at that fixed K.

## How To Run
- Mutual Information:
  - `make select CONFIG=configs/pytorch_default.yaml METHOD=mi`
- L1 Logistic:
  - `make select CONFIG=configs/pytorch_default.yaml METHOD=l1`
- Useful flags:
  - `--target_coverage 0.98` to relax required coverage.
  - `--missingness_threshold 0.5` to drop high‑missing features up front.
  - `--max_features 50` to cap size.
- `--outdir selection_runs` to point at a custom root for artifacts.
  - `--run-name my_selector` to force a specific run folder name.

Artifacts are saved under `selection_runs/run_<timestamp>_select/<method>/`:
- `*_results.json`: selected feature list, full AUC, incremental steps.
- `*_ranking.csv`: full ranking of features with scores.
- `*_auc_curve.png`: AUC vs number of selected features (with full and target lines).
- The run root stores `config_resolved.yaml` and `summary.json` for provenance.

## Applying the Result
1) Open the saved JSON and copy `selected_features`.
2) Paste into `data.features` in your YAML config.
3) Train with the updated config and compare metrics/curves to the full set (e.g., `make train CONFIG=configs/pytorch_default.yaml`).

## Reproducibility
- Time split and preprocessing match training.
- Random seeds set through sklearn APIs; keep `split.random_state` fixed for repeatability.
- Selection does not use W&B; all outputs are local files under `selection_runs/`.

## Extensions (optional improvements)
- Validation‑based stopping: pick K on a validation slice within the training period, then report test once.
- L1 sweep: evaluate several `C` values (e.g., 0.03, 0.1, 0.3, 1.0) and choose by validation AUC with a size penalty.
- Interaction‑aware methods: tree‑based importances or sequential forward selection with time‑aware validation.
- Stability analysis: bootstrap or time‑blocked resampling to assess how often a feature is selected.

## Thesis Writing Tips
- Include the AUC curve figure to show diminishing returns.
- Report the full-features baseline vs selected subset: AUC, AP, confusion stats at the chosen threshold.
- Describe the method in 2–3 sentences (as above) and state your stopping rule and split protocol clearly.

## Roadmap: Neural Feature Selection & Engineering

This plan captures the agreed direction for expanding feature selection into a core experimentation surface, with emphasis on neural methods and tighter coupling to feature engineering.

### Current Baseline (as of 2025-09)
- CLI exposes only mutual information and sparse logistic selectors that evaluate subsets with a logistic baseline (`src/cli/select.py`, `src/selection/mi_selection.py`, `src/selection/l1_selection.py`).
- Selection artefacts live in timestamped run folders with consistent method subdirectories (`selection_runs/run_<timestamp>_select/mi`, `.../l1/`).
- Engineered features (`credit_history_length`, `income_to_loan_ratio`, `fico_avg`, `fico_spread`) are generated during data loading and passed to training, but selectors currently ignore them because they filter strictly to `data.features` from YAML.
- Neural networks consume fixed subsets but do not influence subset discovery beyond post-hoc W&B metrics and SHAP exports.

### Gaps to Close
- No unified abstraction for selectors, forcing bespoke scripts for each new method.
- Duplicate preprocessing/evaluation logic scattered across selector modules rather than reusing the training pipeline helpers.
- Lack of switchable feature-engineering controls in selection runs, making it impossible to quantify engineered feature lift.
- Artefact schema does not support automated comparisons or downstream ingestion.

### Architectural Upgrades
- Introduce a selector base class/registry under `src/selection/` (e.g., `SelectionRunner`) that standardises dataframe access, preprocessing, scoring, and artefact logging. Migrate MI/L1 to the new API first.
- Reorganise selectors into subpackages (`filter/`, `wrapper/`, `neural/`) and move shared helpers such as subset evaluation into `src/selection/utils.py` to enforce consistent train/validation/test handling.
- Extend configs with a `selection` block describing method type, estimator presets, budgets, and regularisation so CLI/Make targets can launch complex selectors via configuration instead of ad-hoc flags.
- Emit a standard artefact set (`selection_summary.json`, `feature_scores.parquet`, `subset_metrics.json`, `plots/…`) under timestamped folders so training runs can ingest selections programmatically and W&B logging can be toggled on when needed.

### Neural-Centric Experiment Tracks
- **Hard-Concrete / L0 Gated MLPs:** Learn per-feature gates jointly with the classifier, penalise expected subset size, and freeze masks for evaluation.
- **Stochastic Gate (STG) Selectors:** Train a lightweight network that samples Bernoulli masks with straight-through gradients, ranking features by learned inclusion probabilities.
- **Autoencoder-Based Screening:** Pretrain autoencoders or bottleneck probes on the training slice to derive nonlinear importance scores before verifying with the production MLP.
- **Wrapper Search with Truncated Training:** Run sequential forward selection or RL-style mask proposals where each candidate subset is scored by a short, warm-start PyTorch training loop.
- **Gradient Attribution Ranking:** Use integrated gradients / DeepLIFT on trained MLPs to generate importances, then feed the scores into the existing coverage-based stopping rule.
- Bake stability analysis into all neural selectors via bootstrapped or time-sliced resampling to track mask overlap across vintages.

### Feature Engineering Integration
- Surface engineered feature groups explicitly in config (e.g., `selection.include_engineered` or dedicated group lists) so selectors can opt into ratios and derived signals while keeping reproducibility explicit.
- Mirror that switch in training by passing `engineered_candidates=[]` to `resolve_feature_inputs` when running “raw-only” baselines.
- Record provenance (raw vs engineered) in selection artefacts/masks to understand which engineered columns survive neural selection pressure.
- Plan paired experiments: run selectors and downstream training with engineered features toggled on/off, logging the subset composition, validation, and test deltas side by side.

### Automation & Tracking
- Add Makefile targets for neural selectors and sweeps (`make select METHOD=nn:concrete`, `make select-grid …`) that resolve to the new registry.
- Store selection metadata in run folders (`local_runs/<id>/selection.json`) to document which mask powered each training run.
- Enable optional W&B logging for selection metrics (mask sparsity, validation curves) to align experimentation histories with training dashboards.
- Expand pytest coverage to include mask determinism, gating regularisers, artefact schema validation, and engineered-feature toggles.

### Immediate Next Steps
1. Prototype the selector base class and migrate MI/L1 onto it, keeping behaviour identical while unifying artefact outputs.
2. Introduce configuration flags for engineered-feature inclusion and verify selectors honour them.
3. Implement a Hard-Concrete gated MLP proof-of-concept on the 10k sample, comparing its subset and metrics to the current L1 baseline.
4. Update documentation and configs (`docs/feature_selection/`, `configs/selection/`) with usage instructions, example presets, and reporting templates.
