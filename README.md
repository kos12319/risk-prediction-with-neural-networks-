# Credit Risk Prediction with Neural Networks and Feature Subset Selection

## Overview
- Build a neural-network–based credit risk model and select a compact, high‑value feature subset.
- Configurable, reproducible pipeline extracted from notebooks.
- All local artifacts write to a single gitignored folder per run under `local_runs/`.

## Dataset (LendingClub)
- This project uses the LendingClub consumer installment loans dataset (2007–2018 vintages).
- Two public files exist: “accepted” loans (funded applications with final statuses) and “rejected” applications (declined, limited covariates).
- Labels are derived from funding outcomes (e.g., Charged Off vs Fully Paid). For recent vintages, be mindful of right‑censoring when interpreting “non‑defaults”.
- Modeling at origination strictly excludes post‑event fields (payments, recoveries, last_* dates, hardship/settlement) to prevent leakage.
- Default positive class convention in configs is `eval.pos_label: 0` (Charged Off); curves/metrics reflect this unless changed.

## Project Layout
- `data/`
  - `raw/`
    - `archives/` — original downloads and compressed files (tracked via Git LFS)
      - `kaggle_accepted_2007_to_2018Q4.csv.gz`, `kaggle_rejected_2007_to_2018Q4.csv.gz`
      - `full data set.zip` (original full‑dataset archive)
    - `full/` — canonical unzipped datasets (ignored by git)
      - `thesis_data_full.csv` (accepted loans, 2007‑06 → 2018‑12)
      - `kaggle_accepted_2007_to_2018Q4.csv`, `kaggle_rejected_2007_to_2018Q4.csv`
    - `samples/` — small CSVs for quick runs (tracked; the 100k CSV is ignored)
      - `thesis_data_sample_100.csv`, `thesis_data_sample_1k.csv`, `thesis_data_sample_10k.csv`
      - `thesis_data_sample_100k.csv` (gitignored) and `thesis_data_sample_100k.zip` (LFS‑tracked)
  - `processed/` — optional cached splits (ignored by git)
- `local_runs/` (gitignored) — per‑run folders with all artifacts
- `configs/`
  - `pytorch/` — PyTorch-only bases (e.g., `base.yaml`) plus optional backend presets
  - `h2o/` — H2O-only bases (e.g., `base.yaml`) plus AutoML presets
  - `pytorch_default.yaml` — PyTorch baseline extending `pytorch/base.yaml`
  - `h2o_default.yaml` — H2O AutoML baseline extending `h2o/base.yaml`
  - `default_automl.yaml` — compatibility shim extending `h2o_default.yaml`
  - `provider_agnostic.yaml` — PyTorch with lender-agnostic features
  - `provider_aware.yaml` — PyTorch including pricing/scoring fields
  - `pytorch_instances/` and `h2o_instances/` — your local presets (gitignored)
- `src/`
  - `data/` — loading, cleaning, splitting
  - `features/` — preprocessing and feature engineering
  - `models/` — neural network (PyTorch)
  - `eval/` — metrics and plots
  - `training/`
    - `base_pipeline.py` — shared, backend-agnostic utilities for data prep/eval (no backend branching)
    - `backends/pytorch/` — PyTorch-specific orchestration (owns run naming, tags, extras)
    - `backends/h2o/` — H2O AutoML orchestration (owns run naming, tags, extras)
    - `train_nn.py`, `train_h2o.py` — backend-specific trainers
    - Backend schemas: `backends/pytorch/schema.py`, `backends/h2o/schema.py` validate backend-only options via Pydantic.
  - `cli/` — command-line entry points
- `docs/exploration/` — exploratory notebooks and reports

## Local Artifacts
- All new runs save to `local_runs/run_YYYYMMDD_HHMMSS/` (gitignored).
- Legacy `reports/` and `models/` are deprecated and ignored.
- Each run folder contains model, metrics, figures, config snapshot, provenance, and optionally a `wandb/` subfolder with downloaded W&B data.

### Run Catalog
- Build a lightweight catalog for comparison and dashboards:
  ```bash
  make run-catalog RUNS_ROOT=local_runs
  ```
- Outputs `local_runs/_catalog.json` with one entry per `run_*` folder, including basic metrics, confusion, data manifest summary, model files, and figure names. CV runs include a link to `cv_metrics.json` when present.
- Render a simple Markdown report for quick browsing:
   ```bash
   make run-catalog-report RUNS_ROOT=local_runs
   ```
   This writes `local_runs/index.md` with grouped runs, AUC, ΔAUC vs previous run in the same group, threshold, and links to figures. It also embeds a small AUC trend plot per group under `local_runs/index_plots/` when `matplotlib` is available.

## Documentation
- Agent Guide: `AGENTS.md`
- ADRs: `docs/architecture/ADRs/` (legacy index at `docs/ADRs/`)
- Note: Unified training/dryrun CLI entries are deprecated; use Make targets or backend-specific CLIs.

## Docs & Architecture Governance
- Single-source flow:
  - Decisions (ADRs): `docs/architecture/ADRs/` (accepted, proposed, rejected).
  - Changes (Journal): `docs/architecture/journal/`.
  - Compiled spec (generated): `docs/architecture/PLATFORM_SPEC.md`.
- Commands:
  - Create entry: `make docs-journal-new TITLE="<title>" [TAGS="..."] [ADRS="0001,0013"]`
  - Build spec: `make docs`
  - Clean spec: `make clean-docs`
- Policy (must-follow for architecture changes):
  - Introduce or modify architecture only alongside an ADR (proposed→accepted) under `docs/architecture/ADRs/`.
  - Log the change with a dated Journal entry referencing the ADR(s).
  - Keep docs Makefile-first; examples should use `make` targets, not raw `python -m`.
  - The compiled spec is gitignored; rebuild with `make docs` when needed.
- Pain Points: `docs/architecture/PAIN_POINTS.md` (active high-priority focus and links to archive and refactor plan)
- Data dictionary: `docs/data/COLUMN_DICTIONARY.md`
- Regenerate column dictionary:
  ```bash
  make venv
  . .venv/bin/activate
  python -m src.cli.gen_column_dict --config configs/pytorch_default.yaml  # or use --csv
  ```

## Dry Run
- PyTorch pipeline smoke test (no artifacts persisted):
  ```bash
  make dryrun CONFIG=configs/pytorch_default.yaml
  ```
- H2O AutoML smoke test (no artifacts persisted):
  ```bash
  make dryrun-h2o AUTOML_CONFIG=configs/h2o_default.yaml
  ```
- Temporal CV smoke test (PyTorch, 2 folds, fast, no artifacts persisted):
  ```bash
  make dryrun-cv
  ```
- Temporal CV smoke test (H2O AutoML, 2 folds, fast, no artifacts persisted; requires Java):
  ```bash
  make dryrun-h2o-cv
  ```
- Both commands write artifacts to a temporary directory that is deleted on exit; a JSON summary is printed to stdout.

## Testing
- Run targeted tests with pytest:
  ```bash
  . .venv/bin/activate
  pytest -q
  ```
- Coverage includes evaluation thresholding, time-split monotonicity, backend wiring, and temporal CV aggregation. In particular, `tests/test_cv_artifacts.py` verifies that CV runs write `cv_metrics.json` with the expected minimal schema.

## Quick Start
1) Create venv and install deps (Python 3.12 preferred):
   ```bash
   make venv
   # or
   python3.12 -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   ```
   Notes:
   - CPU-only environments: if Torch wheel resolution fails or attempts CUDA, install a CPU wheel explicitly (e.g., pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu) before syncing other deps.
   - Apple Silicon: set VECLIB/OMP env already handled; use `make cpu-train` if you hit BLAS thread errors.
2) Choose a config and set dataset path:
   - Provider-agnostic (default): `configs/provider_agnostic.yaml` (excludes int_rate/grade/sub_grade/installment and funded_amnt)
   - Provider-aware: `configs/provider_aware.yaml` (includes pricing/scoring fields)
   - Put your own variants under `configs/pytorch_instances/` (gitignored)
   - Set `data.csv_path` to a CSV (e.g., `data/raw/samples/thesis_data_sample_10k.csv` or `data/raw/full/thesis_data_full.csv`)
   - Outlier handling is configured via `data.winsorize`; toggle with `data.winsorize_enabled` (default true). Listed numeric features are winsorized (quantile/absolute caps) on the training split before scaling.
3) Login to W&B from env (optional, needed for downloads):
   ```bash
   export WANDB_API_KEY=...    # required to pull/download
   export WANDB_ENTITY=your_entity
   # optional: export WANDB_PROJECT=loan-risk-mlp
   make wandb-login
   ```
4) Train the model (PyTorch backend; use `make automl-h2o` for H2O AutoML):
   ```bash
   make train CONFIG=configs/pytorch_default.yaml             # PyTorch training run
   make train CONFIG=configs/pytorch_default.yaml PULL=true   # PyTorch + download W&B files
   # On Linux/WSL or constrained envs, use CPU-only helper:
   make cpu-train CONFIG=configs/pytorch_default.yaml         # PyTorch on CPU with minimal threads
   # Kick off H2O AutoML (defaults to configs/h2o_default.yaml when AUTOML_CONFIG unset)
   make automl-h2o                                    # H2O AutoML pipeline
   make automl-h2o AUTOML_CONFIG=configs/h2o_default.yaml NOTES="grid search"  # custom config
   ```

## Temporal Cross-Validation
- Enable time-aware CV by setting `split.cv.enabled: true` and `split.cv.n_folds` (≥2). The splitter uses an expanding window: the first chunk of data (`split.cv.initial_train_fraction`) seeds the training window, and each fold holds out the next contiguous time block as test data while carving validation from the tail of the training period.
- Additional knobs:
  - `split.cv.mode`: currently `expanding` (default); other modes are rejected until implemented.
  - `split.cv.validation_fraction`: fraction of each fold's training window reserved for validation/threshold selection.
  - `split.cv.gap`: optional row gap between train and test segments to guard against near-term leakage.
  - `split.cv.shuffle_within_folds`: when `true`, shuffles the already time-ordered train/val/test subsets (disabled by default).
- Set `split.cv.train_full_after: true` to automatically fit a final model on the full dataset after CV completes (using the standard single-split pipeline). The final run artifacts live alongside the fold outputs, and the CLI return payload includes both `cv_summary` and final-metric details.
- Artifacts for each fold are written under `run_dir/folds/fold_XX/` with the usual metrics, curves, and manifests. Aggregated metrics and per-fold summaries are saved to `reports/cv_metrics.json` alongside a top-level `README.md` summarizing means/standard deviations. Both the PyTorch (`make train`) and H2O AutoML (`make automl-h2o`) backends honor the cross-validation configuration.
5) Download runs from W&B (to a separate history folder):
   ```bash
   # Pull a specific run (requires a run_id); resolves entity/project from config/env
   make pull-run RUN=entity/project/run_id            # downloads into wandb-history/<run_id>/
   make pull-run RUN=project/run_id                   # uses WANDB_ENTITY from env/config
   make pull-run RUN=run_id                           # uses WANDB_ENTITY and WANDB_PROJECT

   # Pull all runs for the configured project (skips existing folders by default)
   make pull-all                                      # downloads into wandb-history/<run_id>/
   make pull-all FORCE=1                              # force re-download/overwrite
   ```

## Artifacts
- Location: `local_runs/run_YYYYMMDD_HHMMSS/`
- Files:
- Model: `loan_default_model.pt` (PyTorch) or `loan_default_model.zip` (H2O AutoML)
  - Metrics: `metrics.json` (ROC AUC, AP, threshold, classification report)
  - Confusion: `confusion.json` (TP/FP/TN/FN, precision/recall/specificity)
  - Curves: `figures/learning_curves.png`, `figures/roc_curve.png`, `figures/pr_curve.png`
  - Sweeps: `roc_points.csv`, `pr_points.csv`
  - Provenance: `config_resolved.yaml`, `features.json`, `data_manifest.json`, `requirements.freeze.txt`, `training.log`
  - W&B: `wandb.json` with `{id, path, url}`; optional `wandb/` with downloaded files/artifacts (when `PULL=true` or via `pull-run`)

## H2O AutoML
- Switch the backend by setting `model.backend: h2o` in your YAML (see `configs/h2o_default.yaml` for a ready-to-run example that extends the default config).
- Put your own AutoML presets under `configs/h2o_instances/` (gitignored).
- Configure AutoML behaviour via the `automl` block: `progress` (set `true` to re-enable the CLI progress bar), `log_level` (defaults to `WARN` on the JVM side), `suppress_dependency_warnings` (hides repetitive `H2ODependencyWarning` chatter by default), `max_runtime_secs`, `max_models`, `balance_classes`, `include_algos`/`exclude_algos`, `seed`, `nthreads`, `max_mem_size`, and optional `export_checkpoints_dir` and `log_dir`. All inputs are respected by the `make automl-h2o` target.
- On Apple Silicon laptops with 16 GB unified memory the full ~1.5 GB LendingClub CSV fits comfortably; set `automl.max_mem_size` to roughly `8g-10g` so the JVM has headroom while leaving space for macOS and Python. On smaller machines, keep the sample CSV or downsample to avoid garbage-collection churn.
- H2O AutoML requires a functional Java runtime (`java -version`). The training CLI now checks for Java before launching; sandboxed environments that block the JVM will fail fast with guidance.
- H2O's XGBoost backend is available on recent macOS/Apple Silicon with current H2O releases. If logs report `XGBoost is not available; skipping it`, AutoML will proceed without XGBoost models. To force-disable, set `automl.exclude_algos: ['XGBoost']`. If you hit linker/OpenMP issues, install `libomp` (e.g., via Homebrew) and rerun `make automl-h2o`.
- Leaderboard charts now use human-friendly model labels (algorithm + short ID) in both PNG exports and comparison curves. A Pareto-front scatter (`figures/comparison/h2o_pareto_front.png`) and CSV (`h2o_pareto_front.csv`) are emitted when AutoML finishes to highlight accuracy vs latency trade-offs; toggle via `explanation_plots.pareto_front`.
- Feature-importance dashboards now include per-family bar charts (`figures/comparison/per_family_varimp/`) with matching CSVs (`varimp_per_family/`), plus partial dependence plots (PDPs) for the overall leader under `figures/explanations/partial_dependence/`; configure via `explanation_plots.per_family_varimp` and `explanation_plots.partial_dependence` (ICE overlays are not exposed by H2O and are ignored).
- Winners-only variable-importance heatmap readability: control total rows and label sizing via `explanation_plots.{varimp_top_k,varimp_winners_top_k,varimp_winners_sort,varimp_heatmap_row_height,varimp_heatmap_fontsize}`. This caps the feature list globally across category winners and scales figure height so y-axis labels stay legible.
- Preprocessing, train/val/test splits, oversampling, threshold selection, and metric computation follow the same pipeline as the neural-network backend—only the estimator swaps to H2O AutoML under the hood.
- AutoML runs emit the standard artifact set plus an `h2o_leaderboard.csv` and a zipped H2O model (`loan_default_model.zip` by default) inside the run folder for portability.
- Use `AUTOML_CONFIG=...` with `make automl-h2o` to point at alternate configs (e.g., shorter runtimes for smoke tests or vendor-specific feature sets).

## Feature Selection
- How to run (two options):
  - Makefile: `make select CONFIG=configs/pytorch_default.yaml METHOD=mi` (or `METHOD=l1`)
  - Direct:
    ```bash
    python -m src.cli.select --config configs/pytorch_default.yaml --method mi
    python -m src.cli.select --config configs/pytorch_default.yaml --method l1
    ```
- Optional flags (direct invocation): `--target_coverage 0.99 --missingness_threshold 0.5 --max_features 50 --outdir selection_runs --run-name my_selector`
- Outputs under `selection_runs/run_<timestamp>_select/<method>/`:
  - `*_results.json` — selected_features, full_AUC, incremental steps
  - `*_ranking.csv` — full ranking with scores
  - `*_auc_curve.png` — AUC vs number of features
- Run root also contains `config_resolved.yaml` and `summary.json` for provenance.
- Apply the subset:
  1) Open `selection_runs/run_<timestamp>_select/<method>/*_results.json`
  2) Copy `selected_features` into `data.features` in your YAML config (or create a new config variant)
  3) Train with that config and compare to the full set
- Details and method rationale: see `docs/feature_selection/FEATURE_SELECTION.md`.

## Experiment Tracking (W&B)
- Enable in config: `tracking.backend: wandb`; `tracking.wandb.enabled: true`.
- Useful options (see `configs/pytorch_default.yaml`):
  - `run_name` or `run_name_template` — placeholders: `{dataset},{split},{pos},{layers},{nf},{auc},{sha},{run_id}`
  - `group` or `group_template` — default: `{dataset}|{split}|{pos}`
  - `job_type`/`job_type_template`, `tags`/`tag_templates`, `ignore_globs`, `log_artifacts`
- Login via env: set `WANDB_API_KEY` and `WANDB_ENTITY`, then `make wandb-login` or just train (trainer auto‑logins if key is present). Optional `WANDB_PROJECT` overrides config.
- Download W&B data to local folder:
  - After PyTorch training: `make train CONFIG=configs/pytorch_default.yaml PULL=true` → `local_runs/<run_id>/wandb/`
  - Any time: `make pull-run RUN=entity/project/run_id` → `wandb-history/<run_id>/`
- Logged in W&B: per‑epoch loss/val_loss/val_auc/lr/time, final metrics (incl. confusion), env+git metadata, requirements snapshot, figures, interactive confusion matrix panel; key files and model are logged as artifacts.

## Environment Variables
- `WANDB_API_KEY` — required for W&B API login and downloads
- `WANDB_ENTITY` — your W&B user or org (used if not in config)
- `WANDB_PROJECT` — optional, overrides config project for new runs and downloads
- `FORCE_CPU=1` — force CPU training; Makefile `cpu-train` sets this automatically
- Thread controls (set by `cpu-train`): `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1`

## CLI Reference
- PyTorch train: `python -m src.cli.pytorch.train --config CONFIG [--notes TEXT] [--pull] [--cpu]`
- PyTorch dry run: `python -m src.cli.pytorch.dryrun --config CONFIG`
- H2O train: `python -m src.cli.h2o.train --config CONFIG [--notes TEXT] [--pull]`
- H2O dry run: `python -m src.cli.h2o.dryrun --config CONFIG`
- W&B login: `python -m src.cli.wandb_login`
- Pull W&B run: `python -m src.cli.wandb_pull --run ENTITY/PROJECT/RUN_ID [--target DIR] [--config CONFIG]`

## Makefile Targets
- `make train CONFIG=... [PULL=true] [NOTES=...]` (PyTorch backend)
- `make automl-h2o [AUTOML_CONFIG=... PULL=true NOTES=...]`
- `make cpu-train CONFIG=... [PULL=true] [NOTES=...]`
- `make run-catalog [RUNS_ROOT=local_runs] [OUT=path]`
- `make wandb-login`
- `make pull-run RUN=entity/project/run_id` — saves to `wandb-history/<run_id>/`
- `make pull-all [FORCE=1]` — saves all to `wandb-history/<run_id>/`
- `make clean-local-runs` — removes `local_runs/` only
- `make clean-wandb-local` — removes `./wandb` (local SDK logs/cache)
- `make clean-local-history` — removes `./wandb-history` (downloaded run histories)
- `make clean-all-local` — removes `local_runs/`, `selection_runs/`, `./wandb`, and `./wandb-history`
- `make clean-cloud-history FORCE=1` — deletes all runs (and logged artifacts) from the configured W&B project

Note: both backend CLIs perform a lightweight config validation step that checks key invariants (binary target mapping to {0,1}, valid `model.backend`, threshold strategy, and required fields). H2O runs prefer internal class balancing; external oversampling is disabled by default when `model.backend: h2o`.

## Optional: PDF → Markdown Conversion
- Install the extra tooling only when you need PDF conversion support:
  ```bash
  make marker-install
  ```
  This pulls the `marker-pdf` stack from `requirements-marker.txt` into the existing virtual environment.
- Convert a PDF without any external LLM calls (offline by default):
  ```bash
  make marker-pdf MARKER_PAPER=docs/thesis/bibliography/papers/example.pdf \
                   MARKER_PAGE_RANGE=0-4
  ```
  Skip `MARKER_PAGE_RANGE` to process the whole document. Outputs land in `docs/thesis/bibliography/papers_md/` (ignored by git) and large OCR models cache to `~/Library/Caches/datalab/`.

## Dependency Management
- This repo uses pip-tools with a two-file setup:
  - `requirements.in` — human-edited top-level deps (loose pins allowed)
  - `requirements.txt` — compiled, fully pinned lockfile
  - Note on H2O: we pin the H2O Python package and also include a find-links source for the latest stable H2O releases. See the top of `requirements.in` and `requirements.txt` for `--find-links https://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html`. This ensures the Python client and the bundled backend jar are kept in sync with H2O’s stable channel during installs.
- Typical workflow:
  - Edit `requirements.in`
  - Install tools: `make deps-tools`
  - Compile lock: `make deps-compile` (updates `requirements.txt`)
  - Sync venv: `make deps-sync` (installs exactly the pinned set)
  - Alternatively, install directly: `pip install -r requirements.txt`
 - Contributors: prefer editing `requirements.in` and regenerating the lock (avoid hand-editing `requirements.txt`).


## Feature Subset Selection (Scope)
- Goal: identify a minimal subset of origination‑time features with near‑maximal predictive power.
- Approaches: filter (missingness/variance/MI), embedded (L1/logistic, tree importances), wrappers (RFECV/sequential) with time‑aware validation.
- Report performance for all features vs selected subset, and provider‑agnostic vs provider‑aware.

## Notes
- Oversampling applies only to the training split to avoid leakage.
- Engineered features: `credit_history_length` (months from `earliest_cr_line` to `issue_d`), `income_to_loan_ratio`, `fico_avg`, `fico_spread`.
- Post‑origination columns are dropped by default (configurable).

## Notebook Integration
- See `docs/NOTEBOOK_INTEGRATION.md` for mapping from the original notebook to this project, including preserved parts and fixes.
