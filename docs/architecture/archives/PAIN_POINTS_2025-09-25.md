# Pain Points (Archived — 2025-09-25)

This file captures the pain points as of 2025-09-25 before the subsequent reprioritization. See the active list at `docs/PAIN_POINTS.md`.

## High-Level
- [x] **Backend decoupling.** PyTorch and H2O now own dedicated CLI packages (`src/cli/pytorch`, `src/cli/h2o`), backend-specific config stacks (`configs/pytorch/`, `configs/h2o/`), and concrete pipeline subclasses (`PyTorchPipeline`, `H2OPipeline`) that ride on a slim `_run_backend_pipeline` scaffold. Shared code is constrained to data prep/eval helpers so additional backends can plug in without touching PyTorch/H2O logic. Legacy `src/training/pipeline.py` was removed on 2025-09-25 to prevent drift; `make dryrun` (PyTorch) succeeded and `make dryrun-h2o` now fails fast with the Java pre-flight when JVM startup is sandboxed.
  - Implemented in (pre-existing; documented here): `src/cli/pytorch/train.py`, `src/cli/h2o/train.py`, `src/training/backends/`, removal of legacy `src/training/pipeline.py` (see `docs/architecture/journal/2025-09-25-backend-pipeline-separation.md`).
- [x] **Document remaining high-level gaps.** Backlog triage completed on 2025-09-25; medium/low sections below now track concrete follow-ups.
- [x] **H2O Java guardrails.** Added a pre-flight `java -version` check in `train_h2o` so runs fail fast with actionable guidance when Java is missing or sandboxed. `make dryrun-h2o AUTOML_CONFIG=configs/h2o_default.yaml` now surfaces the friendly error (sandbox still blocks JVM startup).
  - Implemented in (pre-existing; documented here): `src/training/train_h2o.py` (function `_verify_java_available`), surfaced via `src/cli/h2o/{train.py,dryrun.py}`.

## Medium
- [x] **Temporal CV orchestrator.** Implemented backend-agnostic temporal CV (expanding window) in `src/training/base_pipeline.py` with per-fold artifacts under `run_dir/folds/fold_XX/` and an aggregated report at `reports/cv_metrics.json`. Added `configs/pytorch/cv_smoke.yaml` and Make target `make dryrun-cv` for a 2-fold smoke test (fast, no artifacts persisted). Verified via `make dryrun-cv`.
  - Implemented in (this changeset exposes usage): `configs/pytorch/cv_smoke.yaml`, `Makefile` (target `dryrun-cv`), `README.md` (Run Catalog and CV smoke test sections). Orchestrator lives in `src/training/base_pipeline.py` (pre-existing; emits `reports/cv_metrics.json`).
- [x] **Selection CLI config resolution.** `src/cli/select.py` reuses `load_config_with_extends` so nested `extends` chains resolve like training. Regression covered by `tests/test_training_config.py`; `make select CONFIG=configs/pytorch_default.yaml METHOD=mi` validated end-to-end.
  - Implemented in (pre-existing; documented here): `src/cli/select.py` uses `load_config_with_extends`.
- [x] **Future backlog triage.** 2025-09-25: reviewed architecture roadmap and promoted the highest-impact follow-ups below; details captured in `docs/architecture/FUTURE_EXTENSIONS.md` and journal entry `2025-09-25-medium-backlog-triage`.
- [x] **Config validation guardrails.** Added lightweight validation in `src/training/config.py` (`validate_and_normalize_config`) invoked by both backends' CLIs (train/dryrun):
  - Verifies `model.backend ∈ {pytorch,h2o}` and requires `data.csv_path`, `data.target_col`, and a binary `data.target_mapping` to {0,1}.
  - Normalizes `eval.pos_label` to {0,1} and checks `eval.threshold.strategy ∈ {fixed,youden_j,f1}`.
  - Ensures `split.method ∈ {time,random}`; when `time`, auto-includes `split.time_col` in `data.parse_dates`.
  - For H2O, discourages external oversampling by default (`oversampling.enabled` coerced to false; prefer AutoML class balancing).
  - Resolves relative CSV paths against the config location and raises clear errors when missing.
  - Implemented in (this changeset): `src/training/config.py` (new `validate_and_normalize_config`), wired into CLIs `src/cli/pytorch/{train.py,dryrun.py}` and `src/cli/h2o/{train.py,dryrun.py}`; documented in `README.md`.
- [x] **Run catalog + artifact manifest.** Added `make run-catalog` (CLI: `python -m src.cli.run_catalog`) to index `local_runs/**/run_*` and emit a compact `_catalog.json` under the runs root. Each entry captures `metrics.json`, `confusion.json`, `data_manifest.json`, detected backend, model files, and figures. CV runs include a pointer to `cv_metrics.json` when present. Verified by training a tiny PyTorch run and generating the catalog.
  - Implemented in (this changeset): `src/cli/run_catalog.py` (new), `Makefile` (target `run-catalog`), `README.md` (Run Catalog section), and `docs/architecture/FUTURE_EXTENSIONS.md` (status updated).

## Low
- [x] **Pytest import ergonomics.** Added `pytest.ini` with `pythonpath=.` to avoid manual `PYTHONPATH` exports when running tests locally.
- [ ] **Open slot.** Reserve for the next low-lift improvement identified during future sweeps.

