# Base Pipeline Refactor Plan

This document details the responsibilities of the shared base pipeline (`src/training/base_pipeline.py`), proposes a decomposition strategy, and outlines a phased plan for incremental refactoring without changing behavior.

## Context
The base pipeline orchestrates end‑to‑end training and evaluation across multiple backends (PyTorch, H2O). It centralizes environment guardrails, data preparation, splitting (single run and temporal CV), training delegation, evaluation/plots, artifacts, and optional W&B logging. Its breadth keeps backends lean but makes the file large and multifaceted.

Source: `src/training/base_pipeline.py`

## Current Responsibilities
- Orchestration
  - `_run_backend_pipeline(...)`: load config, apply env/backends overrides, set up run directories, drive single‑run training or temporal CV, evaluate, and persist artifacts.
  - Backend adapter contract via `BackendPipeline` ABC and dataclasses (`DatasetBundle`, `BackendTrainingResult`, `TrainingRunResult`, `RunContext`).
- Temporal Cross‑Validation
  - `_run_temporal_cv(...)`: expanding‑window fold creation, per‑fold artifact dirs, aggregation, and `reports/cv_metrics.json` emission.
  - `_run_cv_fold(...)`: per‑fold preprocessing, optional train‑only resampling, backend training, evaluation, figures, CSV curves, metrics/confusion, and fold README.
- Evaluation & Artifacts
  - Thresholded evaluation on test at a validation‑selected threshold.
  - ROC/PR figures, ROC/PR CSV points, threshold grid CSV, metrics/confusion JSON, features/manifest JSONs, run/fold README.
- Environment & Reproducibility
  - Safe thread limits, headless plotting, cache dirs; reproducible seeding for Python/NumPy/Torch.
  - Device selection helpers (CUDA/MPS/XPU/CPU) and env overrides.
- Telemetry & Metadata
  - Lightweight system/library/git metadata collection for logging and W&B annotations.
- W&B Integration (optional)
  - Group/job_type templating, run init, metric definitions, summary logging, artifact hints, and backend extension hooks.

## Refactor Goals
- Reduce cognitive load and duplication (e.g., plotting/CSV emission appears in both single‑run and fold flows).
- Improve testability by extracting small, pure helpers with targeted unit tests.
- Constrain blast radius for future features (new CV modes, additional backends, artifact expansions).
- Keep behavior, file/dir layout, and public interfaces stable.
 - Make the common config guardrails strictly backend‑agnostic; enforce backend requirements in each backend pipeline.

## Proposed Decomposition
- Orchestrator Core
  - Move high‑level sequencing into `src/training/orchestrator.py` (keep a shim in `base_pipeline.py` to maintain imports).
- Temporal CV
  - Extract `_run_temporal_cv` and `_run_cv_fold` to `src/training/cv.py`.
- Environment & Device
  - Extract `_apply_common_env_overrides`, `_env_flag`, and `_resolve_torch_device` to `src/training/env.py`.
- Telemetry/Metadata
  - Extract `_collect_system_info` and `_collect_env_metadata` to `src/training/telemetry.py`.
- Evaluation Writer
  - Centralize post‑train evaluation and artifact emission into `src/training/evaluation_writer.py` (metrics/confusion save, plots, ROC/PR CSVs, threshold grid, manifests, READMEs).
- Probability Utilities
  - Extract `_align_probabilities` to `src/training/probability.py`.
- Interfaces
  - Keep `BackendPipeline` ABC and dataclasses in a slim `base_pipeline.py` or move to `src/training/interfaces.py` and re‑export.

## Multi‑Phase Plan (Behavior‑Preserving)
1) Extract Helpers (low risk)
   - Move env flags, device selection, telemetry, and probability alignment to new modules. Import them from `base_pipeline.py` to keep current call sites identical.
   - Add unit tests: env flag parsing, probability alignment, device selection fallbacks (guard with availability checks).

2) Extract Temporal CV — in progress (first cut DONE)
   - Moved `_run_temporal_cv` into `src/training/cv.py` and delegated from `base_pipeline.py` with identical outputs (`cv_metrics.json`, summary README, return payload).
   - Next: relocate `_run_cv_fold` as well and deduplicate plot/CSV emission via a shared evaluation writer.
   - Tests: existing CV smoke tests cover aggregation shape; consider adding unit tests for aggregation math (`roc_auc_mean/std`, `average_precision_mean/std`, confusion sums, `total_test_rows`).

3) Centralize Evaluation Writer — DONE (first cut)
   - Implemented `src/training/evaluation_writer.py` with `write_basic_eval_artifacts` (CV) and `write_full_eval_artifacts` (single run). Both paths now call into this module for plots and JSON/CSV emission.
   - Preserved legacy CSV placement for single runs by copying generated CSVs into `run_dir` for compatibility.
   - Next: consider adding golden-file tests to pin CSV schemas/paths and fold README contents.

4) Thin Orchestrator
   - Move `_run_backend_pipeline` into `orchestrator.py`. Keep `BackendPipeline.run(...)` delegating to the new module. Re‑export for backwards compatibility.

5) Documentation & Cleanups
   - Update internal architecture docs and diagrams once modules settle. Avoid changing CLIs or Make targets.

## Acceptance Criteria
- Identical behavior and artifacts for the same config/seed before vs. after each phase.
- No CLI/Makefile changes. File/dir names and report schemas stay the same.
- Backends unchanged except for imports.

## Risks and Mitigations
- Accidental behavior drift
  - Use “pure extraction” (copy then import), add targeted tests, and compare outputs on a fixed smoke dataset.
- Circular dependencies
  - Keep helpers in leaf modules imported by the orchestrator; avoid helpers importing the orchestrator.
- Churn across PRs
  - Deliver in small, reviewable steps (helpers → CV → evaluation writer → orchestrator move).
