# 2025-09-25 — Backend pipeline separation

## Context
- High-level pain point called for deeper separation between PyTorch and H2O backends so a third backend can slot in cleanly.
- Previous refactors left both backends sharing a monolithic `execute_pipeline` helper and flat CLI modules.

## Decision / Change
- Introduced `BackendPipeline` base class in `src/training/base_pipeline.py` with a reusable `_run_backend_pipeline` core and backend-specific overrides.
- Implemented concrete `PyTorchPipeline` and `H2OPipeline` classes that encapsulate validation, model path resolution, training, and W&B hooks per backend.
- Split CLI surface into backend-specific packages (`src/cli/pytorch`, `src/cli/h2o`) and updated Makefile targets to call them directly. Legacy entry points remain as thin wrappers for compatibility.

## Rationale
- Explicit subclass hooks simplifies reasoning about per-backend behavior and keeps shared code limited to data prep/evaluation scaffolding.
- Dedicated CLI namespaces remove accidental coupling and make it obvious where a new backend should add commands.

## Follow-ups
- Document `BackendPipeline` contract in developer docs and add template/tests to guide new backend implementations.
- Continue shrinking `_run_backend_pipeline` so only data/eval utilities live in the shared layer.

## Validation
- `make dryrun CONFIG=configs/pytorch_default.yaml` (PyTorch) — success.
- `make dryrun-h2o AUTOML_CONFIG=configs/h2o_default.yaml` — stops after the Java pre-flight with the expected sandbox error message (confirms guardrail).
- Removed the legacy `src/training/pipeline.py` module to avoid future regressions and duplicate entry points.
