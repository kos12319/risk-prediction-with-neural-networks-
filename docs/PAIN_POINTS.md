# Pain Points and Recommendations

This document captures friction points identified while reviewing the repo, with quick recommendations to address each.

## Reproducibility & Environment
- Unpinned dependencies. `scikit-learn` API used (`OneHotEncoder(sparse_output=...)`) requires ≥1.2, but `requirements.txt` is unpinned. Pin known‑good versions (e.g., sklearn ≥1.2), and consider a lock or constraints file.
- PyTorch install friction. Bare `torch` in requirements can pull GPU wheels or fail on some OSes. Pin a CPU wheel or document install matrix.
- Missing global seeding. Training, sampling, and splits are not fully deterministic. Add a `set_seed()` that seeds `random`, `numpy`, and `torch` (incl. DataLoader workers) and pass a `torch.Generator` where applicable.

## Config & Data Loading
- Date columns not guaranteed to load. `load_and_prepare()` only includes `features + [target]` in `usecols`; `parse_dates` columns may be omitted, breaking time split and feature engineering (src/data/load.py). Include `parse_dates` in `usecols`.
- Selection CLI ignores `extends`. `src/cli/select.py` loads YAML directly and does not resolve `extends`, unlike training. Harmonize by reusing the training config loader.

## Modeling & Training
- PyTorch validation split nondeterminism. `torch_random_split` is called without a seeded generator, so val fold varies across runs. Provide a `Generator().manual_seed(random_state)`.
- Sparse→dense conversion via `.todense()`. Use `.toarray()` to avoid `numpy.matrix` quirks; ensure arrays are `float32` for frameworks.
- Slightly misleading docstring in `time_based_split` mentions quantiles; code does index split. Consider clarifying the docstring or implementing quantile thresholding.
- Oversampling before validation split. The pipeline oversamples the entire training set and then carves a validation subset from it. Duplicated minority samples can land in both train and val, inflating validation signal and weakening early stopping. Split train→(train_sub, val_sub) deterministically first, then apply oversampling only to train_sub. Prefer class weighting (`training.class_weight`) or focal loss for this NN to avoid prior distortion and calibration drift; if keeping ROS, cap the effective ratio (e.g., ≤1:1) and seed ROS for reproducibility.

## Evaluation & Reporting
- Unconditional note in run README. The summary note says “Evaluated defaults as the positive class.” regardless of `eval.pos_label`. Make this conditional to the actual setting.
- Absolute paths in run README. Artifacts are listed with absolute paths, reducing portability. Prefer paths relative to the repo root or the run directory.
- Threshold sanity. No warnings when the chosen threshold yields degenerate precision/recall. Add a check and brief note in README when metrics are extreme.
- Threshold chosen on the test set. The operating point (Youden/F1/fixed) is selected using test labels and scores, which lets the test set influence the reported operating metrics. Prefer choosing the threshold on validation (or via temporal CV) and reporting its performance on the untouched test set.

## Feature Selection
- Fragile OHE group mapping. Mapping encoded columns back to original features splits on the first underscore, which breaks for names like `fico_range_low` and categories containing underscores (src/selection/utils.py). Build the mapping from the OneHotEncoder internals (`categories_`) and the original `feature_names_in_`, or parse `get_feature_names_out` with the `ColumnTransformer` structure rather than string splits.
- Single holdout evaluation. Incremental AUC uses one train/test split (time or random). Estimates can be unstable. Consider temporal CV (rolling windows) or repeated holdout for more robust curves.

## Repo Hygiene & Tests
- Large data artifacts in git. CSVs/zips under the repo contradict the guidance; move large data outside git and reference via config. Add to `.gitignore`.
- Stray/broken script. `csv_prep.py` calls `pd.read_excel()` without arguments; remove or fix with explicit inputs.
- No tests. Add minimal `pytest` coverage for: preprocessing invariants (no leakage; consistent feature counts), time‑split correctness, and model I/O (save/load works).

## Minor/Edge Cases
- `credit_history_length` can be negative if dates are out of order; clamp to non‑negative or drop anomalies.
- Document that oversampling applies only to training to prevent leakage (already implemented) and that class weight mode (`training.class_weight`) is supported in the backend.
