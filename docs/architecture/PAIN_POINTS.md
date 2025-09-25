# Pain Points (Active — High Priority Focus)

This list reflects the current high‑priority focus. Previous items have been archived at `docs/architecture/archives/PAIN_POINTS_2025-09-25.md`.

## High Priority
Verification (2025-09-25 22:02Z): Re‑verified tiny end‑to‑end checks confirming decoupled backends and guardrails are healthy on `data/raw/samples/thesis_data_sample_1k.csv`.
- PyTorch CV smoke: `make dryrun-cv` → 2 folds completed, AUC≈0.74 mean.
- H2O CV smoke: `make dryrun-h2o-cv` → 2 folds completed (Java OK). Metrics are not tuned (smoke only) but pipeline green.
- Template dry run: `make dryrun-template` → completed with artifacts in temp dir.
Run catalog previously validated via `make run-catalog` + `make run-catalog-report` (requires local_runs). Backend separation remains clean (no shared training CLI, backend-owned schemas/pipelines).
- Backend decoupling (CLI/config/pipeline)
  - [done] Separate backend CLIs and configs; deprecate unified training/dryrun CLI.
    - Change: `python -m src.cli train|dryrun` now exits with guidance. Use `make train`/`make automl-h2o` or explicit modules `src.cli.pytorch.*` / `src.cli.h2o.*`.
    - Docs updated: removed references to `configs/default.yaml` and unified CLI; switched to Makefile-first and backend-specific examples.
    - Effect: Clear separation enables adding a third backend without touching others; users pick backend explicitly via Makefile/CLI.
    - Verification: Ran `make dryrun`, `make dryrun-cv`, `make dryrun-h2o`, and `make dryrun-h2o-cv` on sample CSVs; all completed successfully with metrics and figures.
    - Caveat: Old notebooks or scripts calling `python -m src.cli train` will need to update; the error message points to replacements.
  - [done] Introduce stable backend interfaces and update imports.
    - Change: Added `src/training/interfaces.py` as a façade exposing `BackendPipeline`, `DatasetBundle`, `BackendTrainingResult`, and `RunContext`. Backends now import from `src.training.interfaces` instead of `base_pipeline`.
    - Effect: Decouples backend modules from the shared pipeline implementation; future pipeline moves won’t require touching backend code.
    - Docs: Added `docs/architecture/INTERFACES.md` documenting the contract and extension hooks.
    - Verification: Ran `make dryrun` and `make dryrun-cv` (PyTorch) and `make dryrun-h2o-cv` (H2O; Java present). All completed successfully.
    - Caveat: This is an import‑path change only; the underlying class definitions still live in `base_pipeline` for now and are re‑exported by the façade. A future step can migrate definitions fully without breaking imports.
  - [done] Remove hard requirement for `model.backend` in H2O configs when using the H2O CLI.
    - Change: H2O schema now accepts missing/empty `model.backend` (treated as H2O). Presets (`configs/h2o_default.yaml`, `configs/h2o/cv_smoke.yaml`) no longer set it explicitly.
    - Effect: Further decouples backend selection from shared configs; backend is implied by the entrypoint, making it easier to add a third backend without cross‑coupling.
    - Verification: Ran `make dryrun-h2o-cv` on sample CSV with Java available; completed successfully with metrics and CV artifacts.
    - Caveat: When invoking backend pipelines programmatically (not via the CLI), keep `model.backend` set for clarity.
- Tests and evaluation invariants
  - [done] Added pytest for threshold selection and `pos_label` handling; see tests/test_eval_thresholds.py. Existing tests cover time‑split monotonicity.
  - [done] Validate temporal CV aggregation schema and artifact layout. Added tests/test_cv_artifacts.py to assert `cv_metrics.json` presence and minimal schema under CV smoke runs.
  - [done] Determinism and train‑only oversampling checks. Added tests/test_repro_and_oversampling.py:
    - `test_determinism_same_seed_same_metrics` runs two tiny single runs with identical seeds and asserts metrics.json equality.
    - `test_oversampling_train_only_manifest` asserts that resampling only alters training labels (before/after counts recorded) and leaves validation/test distributions unchanged.
    - Caveat: exact determinism is validated on CPU; accelerator backends (CUDA/MPS) can introduce nondeterminism unless deterministic ops are fully enforced.
  - [done] Backend config guardrails wording aligned with tests.
    - Change: PyTorch schema now raises “PyTorch pipeline requires model.backend to be 'pytorch' (or omitted)” to match `tests/test_training_backends.py`.
    - Verification: `pytest -q` passes locally; message regex matches.
    - Caveat: H2O uses analogous wording; keep messages stable as tests may pin regex substrings.
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

- Makefile portability
  - [done] Fix template backend targets to avoid non-portable `$(or ...)` expansion.
    - Change: `make train-template` and `make dryrun-template` now rely on CLI defaults when `CONFIG` is unset, and only pass `--config` when explicitly provided.
    - Effect: Works across Make variants (GNU/BSD) and avoids accidental mis-parsing that could select the wrong backend.
    - Verification: Ran `make dryrun-template` with no `CONFIG` (uses `configs/template_default.yaml`) and observed a successful dry run on the 1k sample.
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
  - [done] Allow backends to override group naming via `format_group_name`.
    - Change: Added a `format_group_name(base_context, cfg)` hook to `BackendPipeline`. The shared layer now consults the backend first, then falls back to `wandb.group_template` and finally a default `{dataset}|{split}|{pos}|{backend}`.
    - Effect: Removes the last bit of naming policy from the shared pipeline, enabling fully backend-owned grouping semantics for both local runs and W&B.
    - Verification: Ran `make dryrun` on sample config; grouping preserved and run executed successfully.
    - Caveat: Backends that don't override the hook keep current behavior; no breaking changes.
  - Scope: see `docs/architecture/PIPELINE_REFACTOR_PLAN.md` for responsibilities, decomposition ideas, and a multi‑phase plan.
  - [done] Phase 2 — introduce thin per-backend orchestrators that call shared, backend-agnostic utilities, allowing `base_pipeline.py` to shrink further or become purely a library. This preserves common functionality while making a third backend fully plug-in with zero edits to shared code.
    - Change: Added backend-owned runner modules: `src/training/backends/pytorch/runner.py` and `src/training/backends/h2o/runner.py`. Public entrypoints (`src.training.backends.<backend>.train_from_config`) now resolve via these runners instead of the `pipeline` modules.
    - Effect: Decouples external interface from the shared base pipeline and establishes backend-owned orchestration seams for future extraction.
    - Verification: Executed `make dryrun` on sample CSV; run completed successfully with unchanged metrics and artifacts.
    - Caveat: This is a no-functional-change structural step; orchestration still routes through the shared `_run_backend_pipeline` until subsequent phases migrate responsibilities out of `base_pipeline.py`.
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

  - [done] Centralize evaluation writer
  - Change: Introduced `src/training/evaluation_writer.py` with `write_basic_eval_artifacts` (CV folds) and `write_full_eval_artifacts` (single runs). Replaced duplicated plotting/metrics emission in `base_pipeline.py` and `cv.py` with calls into the new module. For single runs, CSV artifacts (roc_points/pr_points/threshold_metrics) are still available under `run_dir` for compatibility.
  - Effect: Eliminates duplication and tightens separation: shared evaluation output is backend-agnostic, reducing pipeline size and easing future backend additions.
  - Verification: Ran `make dryrun` and `make dryrun-cv` on sample CSV; metrics/figures/JSONs produced as before. Paths for CSVs remain compatible for single runs; CV fold artifacts unchanged.
  - Caveat: None; functionality is a pure extraction. If the central writer import fails, both paths fall back to the previous inline writers.

  - [done] Introduce orchestrator shim to decouple callers from implementation.
    - Change: Added `src/training/orchestrator.py` with `run_backend_pipeline(...)` delegating to the legacy implementation. Updated `BackendPipeline.run(...)` to call the shim.
    - Effect: Stable import path for orchestration enables a safe, stepwise move of the heavy `_run_backend_pipeline` out of `base_pipeline.py` without breaking callers.
    - Verification: Ran a tiny PyTorch dry run (`make dryrun`) on the 1k sample; metrics and artifacts emitted successfully.
    - Caveat: Implementation still lives in `base_pipeline.py`; a follow-up will complete the move once parity is re-confirmed across smoke tests.

## Notes
- We will gate deeper refactors behind tests to pin behavior and artifacts. CV smoke tests must remain fast and artifact‑light.
  - [done] Route H2O training via backend-owned module.
    - Change: `H2OPipeline` now imports `train_h2o` from `src/training/backends/h2o/train.py`.
      The current implementation remains in place but is re-exported through the backend
      namespace to decouple callers from legacy paths and prepare for a future move.
    - Effect: Further separation of backends; future H2O changes won’t touch shared modules.
    - Verification: Imported `H2OPipeline` successfully and ran `make dryrun-cv` (PyTorch)
      to ensure unrelated paths remain green. H2O smoke runs still require Java; behavior
      unchanged.
    - Caveat: In environments without Java, H2O runs still fail fast by design.
