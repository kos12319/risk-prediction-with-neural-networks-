# Pain Points (Active — High Priority Focus)

This list reflects the current high‑priority focus. Previous items have been archived at `docs/architecture/archives/PAIN_POINTS_2025-09-25.md`.

## High Priority
- Tests and evaluation invariants
  - [progress] Added pytest for threshold selection and `pos_label` handling; see tests/test_eval_thresholds.py. Existing tests cover time‑split monotonicity. Determinism and train‑only oversampling checks remain.
  - [todo] Validate temporal CV aggregation schema and artifact layout (`reports/cv_metrics.json`, per‑fold files).
- H2O CV parity and ergonomics
  - [done] Added an H2O temporal CV smoke‑test preset and Make target.
    - Config: `configs/h2o/cv_smoke.yaml` (2 folds, ~15s runtime budget).
    - Target: `make dryrun-h2o-cv` (mirrors PyTorch `make dryrun-cv`).
    - Caveat: requires a working Java runtime (`java -version` must succeed). In sandboxed environments, this will fail fast with an actionable error.
  - Ensure AutoML class balancing and guardrails remain consistent under CV.
- Run catalog usability
  - Generate lightweight HTML/Markdown summaries from `_catalog.json` and add comparison helpers (delta tables, trends over time).
  - Wire simple dashboards into docs for quicker review cycles.
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

## Notes
- We will gate deeper refactors behind tests to pin behavior and artifacts. CV smoke tests must remain fast and artifact‑light.
