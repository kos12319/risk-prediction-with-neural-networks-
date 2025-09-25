# Pain Points (Active — High Priority Focus)

This list reflects the current high‑priority focus. Previous items have been archived at `docs/architecture/archives/PAIN_POINTS_2025-09-25.md`.

## High Priority
- Tests and evaluation invariants
  - [progress] Added pytest for threshold selection and `pos_label` handling; see tests/test_eval_thresholds.py. Existing tests cover time‑split monotonicity. Determinism and train‑only oversampling checks remain.
  - [done] Validate temporal CV aggregation schema and artifact layout. Added tests/test_cv_artifacts.py to assert `cv_metrics.json` presence and minimal schema under CV smoke runs.
  - [done] Determinism and train‑only oversampling checks. Added tests/test_repro_and_oversampling.py:
    - `test_determinism_same_seed_same_metrics` runs two tiny single runs with identical seeds and asserts metrics.json equality.
    - `test_oversampling_train_only_manifest` asserts that resampling only alters training labels (before/after counts recorded) and leaves validation/test distributions unchanged.
    - Caveat: exact determinism is validated on CPU; accelerator backends (CUDA/MPS) can introduce nondeterminism unless deterministic ops are fully enforced.
- H2O CV parity and ergonomics
  - [done] Added an H2O temporal CV smoke‑test preset and Make target.
    - Config: `configs/h2o/cv_smoke.yaml` (2 folds, ~15s runtime budget).
    - Target: `make dryrun-h2o-cv` (mirrors PyTorch `make dryrun-cv`).
    - Caveat: requires a working Java runtime (`java -version` must succeed). In sandboxed environments, this will fail fast with an actionable error.
  - Ensure AutoML class balancing and guardrails remain consistent under CV.
- Run catalog usability
  - [done] Generate lightweight Markdown summaries from `_catalog.json`.
    - New CLI: `python -m src.cli.run_catalog_report --runs-root local_runs`.
    - Make target: `make run-catalog-report` (writes `local_runs/index.md`).
    - Usage: run a tiny smoke (`make dryrun-cv` or `make dryrun-h2o-cv`), build catalog (`make run-catalog`), then report.
    - Caveat: figures/links resolve relative to `local_runs/`; ensure runs were created with `output.runs_root` set.
  - [next] Add simple delta tables and trend plots atop the catalog (HTML export).
- Config schema hardening
  - Introduce a stricter schema layer (e.g., Pydantic) on top of the current lightweight validator to catch leakage, mutually exclusive options, and type shape errors earlier.
  - Provide clearer error messages and config linting hooks.
- Base pipeline refactor (phased)
  - Rationale: reduce cognitive load, improve testability, and constrain blast radius when adding features.
  - [progress] Removed backend-specific naming branches from the shared pipeline; run naming is now owned by each backend via `format_run_name`.
    - Implemented default naming in `src/training/backends/pytorch/pipeline.py` to mirror prior MLP-style names.
    - H2O pipeline already provided a backend-specific name; no change needed.
    - Result: `base_pipeline.py` no longer inspects backend types to construct names, reducing coupling.
    - Caveat: If a backend does not implement `format_run_name` and no `run_name_template` is configured, a generic `{dataset}|{split}|{pos}|{backend}|auc{auc}` name is used.
  - Scope: see `docs/architecture/PIPELINE_REFACTOR_PLAN.md` for responsibilities, decomposition ideas, and a multi‑phase plan.
  - [done] Backend-specific config rules moved out of shared validator.
    - Change: `validate_and_normalize_config` is now backend-agnostic (data/split/eval only). Backend CLIs/Pipelines enforce their own requirements via `validate_config`.
    - Effect: decouples config schema across backends and simplifies adding a third backend in the future. H2O oversampling behavior is no longer silently altered by the common layer; presets keep it disabled by default for H2O.
    - Caveat: configs that relied on the implicit H2O oversampling flip must keep `oversampling.enabled: false` (the default in `configs/h2o/base.yaml`).

## Notes
- We will gate deeper refactors behind tests to pin behavior and artifacts. CV smoke tests must remain fast and artifact‑light.
