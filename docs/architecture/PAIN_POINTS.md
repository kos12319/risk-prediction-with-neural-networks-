# Pain Points (Active — High Priority Focus)

This list reflects the current high‑priority focus. Previous items have been archived at `docs/architecture/archives/PAIN_POINTS_2025-09-25.md`.

## High Priority
- Backend decoupling (CLI/config/pipeline)
  - [done] Separate backend CLIs and configs; deprecate unified training/dryrun CLI.
    - Change: `python -m src.cli train|dryrun` now exits with guidance. Use `make train`/`make automl-h2o` or explicit modules `src.cli.pytorch.*` / `src.cli.h2o.*`.
    - Docs updated: removed references to `configs/default.yaml` and unified CLI; switched to Makefile-first and backend-specific examples.
    - Effect: Clear separation enables adding a third backend without touching others; users pick backend explicitly via Makefile/CLI.
    - Verification: Ran `make dryrun`, `make dryrun-cv`, `make dryrun-h2o`, and `make dryrun-h2o-cv` on sample CSVs; all completed successfully with metrics and figures.
    - Caveat: Old notebooks or scripts calling `python -m src.cli train` will need to update; the error message points to replacements.
- Tests and evaluation invariants
  - [done] Added pytest for threshold selection and `pos_label` handling; see tests/test_eval_thresholds.py. Existing tests cover time‑split monotonicity.
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
  - [done] Ensure AutoML class balancing and guardrails remain consistent under CV.
    - Change: H2O backend now validates AutoML options via a backend-scoped schema and applies identical settings across folds. Environment overrides (`H2O_BALANCE_CLASSES`, `H2O_MAX_AFTER_BALANCE_SIZE`, `H2O_CLASS_SAMPLING_FACTORS`) are consistently honored for CV and single runs.
    - Caveat: Java is still required for any H2O run; CV smoke uses low `max_runtime_secs` but may exceed tight CI timeouts on slow machines.
- Run catalog usability
  - [done] Generate lightweight Markdown summaries from `_catalog.json`.
    - New CLI: `python -m src.cli.run_catalog_report --runs-root local_runs`.
    - Make target: `make run-catalog-report` (writes `local_runs/index.md`).
    - Usage: run a tiny smoke (`make dryrun-cv` or `make dryrun-h2o-cv`), build catalog (`make run-catalog`), then report.
    - Caveat: figures/links resolve relative to `local_runs/`; ensure runs were created with `output.runs_root` set.
  - [done] Add delta AUC and simple trend plots.
    - Change: `run_catalog_report` now computes per-group ΔAUC vs previous run and embeds a small AUC trend PNG per group under `local_runs/index_plots/`.
    - Effect: Quickly spot regressions/improvements across iterations within a setup.
    - Caveat: Plots require `matplotlib`; if unavailable, the report still renders tables without images. Sorting prefers run creation time when present; falls back to `run_id`.
- Config schema hardening
  - [done] Introduce a stricter schema layer (Pydantic) for backend-specific configs.
    - Change: Added `src/training/backends/pytorch/schema.py` and `src/training/backends/h2o/schema.py`; pipelines call these to validate backend-only options (e.g., PyTorch `training.class_weight`, H2O `automl.*`). Shared invariants remain in the common validator.
    - Effect: Clearer errors on type/shape issues without coupling backends; paves the way for a third backend.
    - [done] Extend shared guardrails for temporal CV and leakage.
      - Change: Shared validator enforces CV parameters (n_folds≥2, fractions in (0,1), mode='expanding') and requires `data.leakage_cols` when `data.drop_leakage=true`. It already ensures time-based split adds `time_col` to `parse_dates`.
      - Effect: Clearer, earlier failures on misconfigured CV and leakage toggles without backend coupling.
      - Caveat: Bounds are conservative (only checks ranges/types); backend-specific contradictions remain validated by each backend schema.
- Base pipeline refactor (phased)
  - Rationale: reduce cognitive load, improve testability, and constrain blast radius when adding features.
  - [done] Extract shared helpers (Phase 1: low-risk pure moves)
    - Change: Moved probability alignment to `src/training/probability.py` and env overrides to `src/training/env.py` (with a thin compat shim in `base_pipeline.py`).
    - Effect: Reduces base pipeline surface and makes helpers reusable across backends without duplication.
    - Validation: `make dryrun CONFIG=configs/pytorch_default.yaml` passes and emits metrics/figures as before.
    - Caveat: Backends should prefer importing helpers from the new modules; legacy wrappers remain for now.
  - [done] Removed backend-specific naming branches from the shared pipeline; run naming is now owned by each backend via `format_run_name`.
    - Implemented default naming in `src/training/backends/pytorch/pipeline.py` to mirror prior MLP-style names.
    - H2O pipeline already provided a backend-specific name; no change needed.
    - Result: `base_pipeline.py` no longer inspects backend types to construct names, reducing coupling.
    - Caveat: If a backend does not implement `format_run_name` and no `run_name_template` is configured, a generic `{dataset}|{split}|{pos}|{backend}|auc{auc}` name is used.
  - Scope: see `docs/architecture/PIPELINE_REFACTOR_PLAN.md` for responsibilities, decomposition ideas, and a multi‑phase plan.
  - [done] Backend-specific config rules moved out of shared validator.
    - Change: `validate_and_normalize_config` is now backend-agnostic (data/split/eval only). Backend CLIs/Pipelines enforce their own requirements via `validate_config`.
    - Effect: decouples config schema across backends and simplifies adding a third backend in the future. H2O oversampling behavior is no longer silently altered by the common layer; presets keep it disabled by default for H2O.
    - Caveat: configs that relied on the implicit H2O oversampling flip must keep `oversampling.enabled: false` (the default in `configs/h2o/base.yaml`).
  - [done] Decouple backend validation and schemas from the shared CLI/validator.
    - Change: Backend pipelines own schema validation via Pydantic modules; CLIs no longer import the shared validator. The shared pipeline still enforces data/eval/split invariants.
    - Effect: Backends are more self-contained and ready for new entrants without touching shared code.
- [done] Extract temporal CV orchestration to a dedicated module.
    - Change: Moved temporal CV and fold logic into `src/training/cv.py` and delegated from `base_pipeline.py` without behavior changes.
    - Effect: Backends remain independent; the shared pipeline is thinner and easier to test. This is a step toward a slimmer orchestrator.
    - Verification: Ran `make dryrun-cv` and `make dryrun-h2o-cv`; both completed and emitted `cv_metrics.json` and summaries as before.
    - Caveat: Only the orchestration moved; evaluation writer remains embedded and will be extracted in a follow-up phase.

  - [done] Remove implicit PyTorch fallback in CV metadata.
    - Change: CV summary now records `backend` via the active backend instance (`backend.name`) instead of `model.backend` with a `'pytorch'` default.
    - Effect: Eliminates a subtle coupling to PyTorch defaults and ensures correct backend attribution for H2O and any future backends.
    - Verification: `make dryrun` and `make dryrun-cv` completed on sample configs; CV metadata is emitted only for CV runs and now reflects the proper backend.
    - Caveat: None; behavior is unchanged for non-CV runs.

## Notes
- We will gate deeper refactors behind tests to pin behavior and artifacts. CV smoke tests must remain fast and artifact‑light.
