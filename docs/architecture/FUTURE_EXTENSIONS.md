# Future Extensions Roadmap

The refactor introduced new modular boundaries so upcoming enhancements can plug into well-defined extension points. Below are high-value follow-ups that build on the new structure.

## Model & Training Extensions
- **Torch model registry** (`src/models/torch_factory.py`)
  - Add additional architectures (e.g., residual MLP, TabTransformer, TabNet) by registering new builders.
  - Support hyperparameter templates so configs can reference concise model aliases.
- **Resampling strategies** (`src/training/resample.py`)
  - Implement class-distance aware samplers (ADASYN, Borderline-SMOTE) and expose per-strategy configs.
  - Allow sequential resampling (e.g., undersample majority before SMOTE) via a composable pipeline.
- **Cross-validation orchestrator**
  - Build a reusable `KFoldRunner` that uses `train_val_test_split` to emit per-fold data bundles and aggregates metrics.
  - Enable H2O AutoML ensembling by stacking fold models or majority voting.
- **H2O temporal CV integration**
  - Generate chronological fold assignments and pass them to AutoML via a `fold_column` so stacked ensembles can consume out-of-fold predictions without leaking future vintages.
  - Surface config flags to toggle between random CV, temporal CV, and blend-only regimes while keeping evaluation metrics comparable.
- **H2O blending frame support**
  - Extend preprocessing to carve out a configurable blending slice from the training period and wire it into `train_h2o`.
  - Document guidance for selecting recent vintages, ensuring the blender stays disjoint from the final test set, and capturing the resulting artifacts.
- **AutoGluon backend spike**
  - Prototype a Python-native AutoML backend (`train_autogluon`) that mirrors the H2O contract (probabilities, leader metadata, artifact bundle) while respecting Makefile-first orchestration.
  - Benchmark accuracy/runtime vs the current H2O flow on sampled vintages; only invest in full integration if the spike shows clear gains and artifact parity is achievable without excessive maintenance overhead.
  - Document dependency impacts (PyTorch/LightGBM/CatBoost wheels on Apple Silicon) and update CI/install guidance before promoting it beyond experimental status.

## Data & Preprocessing
- **Encoder plugins** (`src/features/preprocess.py`)
  - Introduce advanced categorical encoders (Weight-of-Evidence, Target smoothing, CatBoost). Each should follow the existing registry pattern for plug-and-play use.
  - Add numerical transformer mixins (quantile transformer, power transformer) selectable via config.
- **Feature selectors**
  - Package feature resolution into reusable strategies (e.g., whitelist, blacklist, top-k). The current `resolve_feature_inputs` helper is the entry point.
- **Time-series context features**
  - Use `autogluon.timeseries` (or similar) to forecast portfolio-level metrics (monthly default rate, funding volume) and join leakage-safe aggregates back into loan-level records for downstream tabular training.
  - Ensure aggregate features are generated strictly from historical data available at each `issue_d`, and keep the primary train/val/test split time-based so evaluation remains chronologically valid.
 - **Employment title normalization (job families)**
   - Aggregate free-text `emp_title` into a compact set of job families to capture signal without high-cardinality blow-up or fairness risk.
   - Options:
     - Local: `skrub` SimilarityEncoder or TF–IDF + mini-batch KMeans to cluster titles; label clusters with human-friendly names; persist vectorizer+clustering artifacts for reproducibility.
     - Taxonomy: code titles to SOC/Census/ESCO via external coders (SOCcer, Census I&O), then map to 20–40 families; cache responses to avoid drift.
   - Wire as a mixed encoder: apply this normalization to `emp_title` only; keep one-hot for low-card categoricals; preserve time-split safety (fit on train only) and document bias checks.

## Evaluation & Comparison
- **Binary evaluator enhancements** (`src/eval/binary.py`)
  - Add calibration metrics, KS-statistic, lift charts, and cost curves.
  - Support configurable operating points (e.g., recall targets) alongside threshold sweeps.
- **H2O leaderboard comparison CLI**
  - Create a command (e.g., `make compare-h2o RUNS="run_a,run_b"`) that loads `h2o_leaderboard.csv` artifacts, highlights deltas, and exports summary tables/plots.
- **Run catalog & dashboards**
  - Implement a `RunCatalog` module that indexes `local_runs/**/metrics.json` and `config_resolved.yaml`, enabling filtered reports or notebooks.
- **Ensembling harness**
  - Provide utilities to blend predictions from multiple runs (PyTorch + H2O) and evaluate stacked models via the evaluator module.

## Tooling & DX
- **Config validation**
  - Add Pydantic schemas for config files to validate incompatible options (e.g., SMOTE without numerical features).
- **Artifact manager** (`src/utils/artifacts.py`)
  - Extend the manager to understand artifact “types” (model, metrics, plots) and create signed manifests for reproducibility.
- **Testing**
  - Add targeted unit tests for `train_val_test_split`, preprocessing pipelines, resampler combinations, and evaluator threshold logic using sample datasets under `tests/`.

These tasks intentionally sit atop the new abstractions. Each addition should leverage the small, testable helpers introduced in this refactor, keeping the main pipeline focused on orchestration.
