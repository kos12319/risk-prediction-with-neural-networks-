# Pain Points

## High-Level
- [x] **Backend decoupling.** PyTorch and H2O now own dedicated CLI packages (`src/cli/pytorch`, `src/cli/h2o`), backend-specific config stacks (`configs/pytorch/`, `configs/h2o/`), and concrete pipeline subclasses (`PyTorchPipeline`, `H2OPipeline`) that ride on a slim `_run_backend_pipeline` scaffold. Shared code is constrained to data prep/eval helpers so additional backends can plug in without touching PyTorch/H2O logic.
- [x] **Document remaining high-level gaps.** Backlog triage completed on 2025-09-25; medium/low sections below now track concrete follow-ups.
- [x] **H2O Java guardrails.** Added a pre-flight `java -version` check in `train_h2o` so runs fail fast with actionable guidance when Java is missing or sandboxed. `make dryrun-h2o AUTOML_CONFIG=configs/h2o_default.yaml` now surfaces the friendly error (sandbox still blocks JVM startup).

## Medium
- [x] **Selection CLI config resolution.** `src/cli/select.py` reuses `load_config_with_extends` so nested `extends` chains resolve like training. Regression covered by `tests/test_training_config.py`; `make select CONFIG=configs/pytorch_default.yaml METHOD=mi` validated end-to-end.
- [x] **Future backlog triage.** 2025-09-25: reviewed architecture roadmap and promoted the highest-impact follow-ups below; details captured in `docs/architecture/FUTURE_EXTENSIONS.md` and journal entry `2025-09-25-medium-backlog-triage`.
- [ ] **Config validation guardrails.** Add schema-based validation (e.g., Pydantic) in `src/training/config.py` so incompatible settings (SMOTE on dense-only features, backend mismatches) fail fast before hitting the pipelines.
- [ ] **Temporal CV orchestrator.** Introduce a reusable fold runner that feeds `train_val_test_split` bundles into backend adapters and aggregates metrics for CV + `train_full_after` flows; expose via Makefile targets for both PyTorch and H2O.
- [ ] **Run catalog + artifact manifest.** Build a lightweight indexer over `local_runs/**/` that emits structured manifests (metrics, configs, artifacts) to enable comparisons, dashboards, and provenance checks.

## Low
- [x] **Pytest import ergonomics.** Added `pytest.ini` with `pythonpath=.` to avoid manual `PYTHONPATH` exports when running tests locally.
- [ ] **Open slot.** Reserve for the next low-lift improvement identified during future sweeps.
