# Backend Interfaces

Backends integrate with the training platform via a small set of interfaces that are intentionally stable across refactors.

- Import path: `src.training.interfaces`
  - `BackendPipeline` — abstract contract a backend implements to plug into the shared orchestration.
  - `DatasetBundle` — preprocessed arrays (train/val/test) and feature names.
  - `BackendTrainingResult` — standardized return payload from `BackendPipeline.train(...)`.
  - `RunContext` — run identifiers and artifact manager for file output.

Guidelines:
- Backends must import these symbols from `src.training.interfaces` (not `base_pipeline`). The module is a stable façade; we can relocate internals without touching backend code.
- The shared pipeline remains backend‑agnostic and focuses on: config load/extends, time‑split/CV, preprocessing, resampling isolation, evaluation writing, and artifact management.
- Backend responsibilities include: model config validation, environment overrides, model path resolution, training, optional W&B extras, and backend‑specific run/group naming.

Minimal contract (hooks):
- `validate_config(cfg)` — enforce backend‑specific schema and constraints.
- `apply_env_overrides(cfg)` — optionally adjust env‑driven options (e.g., H2O balancing flags).
- `resolve_model_path(out_cfg, artifact_mgr, fold_meta)` — choose artifact filename/location.
- `prepare_model_config(model_cfg, training_cfg, y_train)` — derive computed options (e.g., class weights).
- `train(dataset, model_cfg, training_cfg, eval_cfg, run_context, model_path, random_seed, pos_label, fold_meta, wandb_run, wandb_enabled, cfg)` — fit a model and return probabilities, label index, model path, and optional raw extras.
- Optional: `log_wandb(...)`, `extra_artifact_lines(...)`, `format_run_name(...)`, `format_group_name(...)`, `additional_wandb_tags(...)`.

See `src/training/backends/template/` for a compact reference implementation.

