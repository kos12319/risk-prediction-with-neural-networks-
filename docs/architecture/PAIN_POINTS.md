# Pain Points (Active — High Priority Focus)

This list reflects the current high‑priority focus. Previous items have been archived at `docs/architecture/archives/PAIN_POINTS_2025-09-25.md`.

## High Priority
- Tests and evaluation invariants
  - Add pytest coverage for: time split monotonicity; train‑only oversampling; validation‑based threshold selection; `eval.pos_label` handling; seeded determinism (Python/NumPy/Torch/DataLoader workers).
  - Validate temporal CV aggregation schema and artifact layout (`reports/cv_metrics.json`, per‑fold files).
- H2O CV parity and ergonomics
  - Add an H2O temporal CV smoke test target (mirrors PyTorch `make dryrun-cv`) and document Java setup options and constraints.
  - Ensure AutoML class balancing and guardrails remain consistent under CV.
- Run catalog usability
  - Generate lightweight HTML/Markdown summaries from `_catalog.json` and add comparison helpers (delta tables, trends over time).
  - Wire simple dashboards into docs for quicker review cycles.
- Config schema hardening
  - Introduce a stricter schema layer (e.g., Pydantic) on top of the current lightweight validator to catch leakage, mutually exclusive options, and type shape errors earlier.
  - Provide clearer error messages and config linting hooks.
- Base pipeline refactor (phased)
  - Rationale: reduce cognitive load, improve testability, and constrain blast radius when adding features.
  - Scope: see `docs/architecture/PIPELINE_REFACTOR_PLAN.md` for responsibilities, decomposition ideas, and a multi‑phase plan.

## Notes
- We will gate deeper refactors behind tests to pin behavior and artifacts. CV smoke tests must remain fast and artifact‑light.

