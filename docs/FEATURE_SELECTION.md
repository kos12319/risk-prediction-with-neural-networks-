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
  - `python -m src.cli.select --config configs/default.yaml --method mi`
- L1 Logistic:
  - `python -m src.cli.select --config configs/default.yaml --method l1`
- Useful flags:
  - `--target_coverage 0.98` to relax required coverage.
  - `--missingness_threshold 0.5` to drop high‑missing features up front.
  - `--max_features 50` to cap size.
  - `--outdir reports/selection` for artifacts location.

Artifacts are saved under `reports/selection/<method>/`:
- `*_results.json`: selected feature list, full AUC, incremental steps.
- `*_ranking.csv`: full ranking of features with scores.
- `*_auc_curve.png`: AUC vs number of selected features (with full and target lines).

## Applying the Result
1) Open the saved JSON and copy `selected_features`.
2) Paste into `data.features` in your YAML config.
3) Train with the updated config and compare metrics/curves to the full set.

## Reproducibility
- Time split and preprocessing match training.
- Random seeds set through sklearn APIs; keep `split.random_state` fixed for repeatability.
- Selection does not use W&B; all outputs are local files under `reports/selection/`.

## Extensions (optional improvements)
- Validation‑based stopping: pick K on a validation slice within the training period, then report test once.
- L1 sweep: evaluate several `C` values (e.g., 0.03, 0.1, 0.3, 1.0) and choose by validation AUC with a size penalty.
- Interaction‑aware methods: tree‑based importances or sequential forward selection with time‑aware validation.
- Stability analysis: bootstrap or time‑blocked resampling to assess how often a feature is selected.

## Thesis Writing Tips
- Include the AUC curve figure to show diminishing returns.
- Report the full‑features baseline vs selected subset: AUC, AP, confusion stats at the chosen threshold.
- Describe the method in 2–3 sentences (as above) and state your stopping rule and split protocol clearly.

