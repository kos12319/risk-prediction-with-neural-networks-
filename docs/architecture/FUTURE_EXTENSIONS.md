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

## Data & Preprocessing
- **Encoder plugins** (`src/features/preprocess.py`)
  - Introduce advanced categorical encoders (Weight-of-Evidence, Target smoothing, CatBoost). Each should follow the existing registry pattern for plug-and-play use.
  - Add numerical transformer mixins (quantile transformer, power transformer) selectable via config.
- **Feature selectors**
  - Package feature resolution into reusable strategies (e.g., whitelist, blacklist, top-k). The current `resolve_feature_inputs` helper is the entry point.

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
