# Thesis Iteration 1 — 1k H2O Runs: Open Issues and Resolutions

Context: local run group `local_runs/thesis_data_sample_1k|time|co|h2o`, W&B project set via config. This file tracks problems seen in the 1k sweep and how we address them.

## 1) Deep Learning models stopped early (budget)
- Cause: `automl.max_runtime_secs` limited search (600s). DL grids were cut when the global runtime expired.
- Fix: Set `automl.max_runtime_secs: 0` in all 1k configs to disable the runtime cap.
  - Files updated: `docs/experiments/suites/thesis_iter1/h2o/1k/*.yaml`.
- Rerun (Makefile‑first):
  - `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/agnostic.yaml`
  - `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/selected.yaml`
  - `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/aware.yaml`
  - `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/1k/selected_plus_providers.yaml`

## 2) Figures show numbers without legend
- Source: H2O leaderboard charts and comparison curves.
- Current behavior (after patch):
  - Bar charts now label models with a short, readable label: `rank. algo [short_id]`.
  - ROC/PR comparison curves include a legend with the same labels.
- Code refs: `src/training/train_h2o.py` — label map and plots.

## 3) `h2o_logs` folder is verbose
- Created by: `h2o.init(log_dir=...)` in `src/training/train_h2o.py`.
- Location: defaults to `<run_dir>/h2o_logs/` (configurable via `automl.log_dir`).
- Level: we set JVM `log_level: WARN` by default, but H2O writes multiple rolling files (INFO/DEBUG from subsystems may still appear). This is expected H2O behavior.

## 4) Multiple leaderboard files (`h2o_leaderboard_*.csv`)
- `h2o_leaderboard.csv`: the standard AutoML leaderboard (validation‑sorted).
- `h2o_leaderboard_extra.csv`: leaderboard with extra columns (when `leaderboard_extra_columns: ALL`).
- `h2o_leaderboard_test.csv`: leaderboard scored on the held‑out test frame.
- Figures source: plots prefer the `test` leaderboard if present; otherwise the default.
- Enhancement (done): Added a new figure for the best model per category (by AUC) to reduce repeated entries per algo.
  - Output: `figures/h2o_best_per_category_auc.png`.

## 5) Empty “Learning Curves” image
- Cause: H2O AutoML has no per‑epoch training history, so the plot was empty.
- Fix: `plot_learning_curves` now detects the absence of history and saves an annotated image (“N/A for this backend”) instead of a blank chart.
- Code: `src/eval/metrics.py`.

## 6) Leaderboard and Pareto front not visible on W&B
- Expected logging: tables/images are logged in `src/training/pipeline.py` when backend is H2O.
  - Tables: `h2o_leaderboard` (if CSV present).
  - Images: `h2o_leaderboard_{auc,logloss,rmse}`, `comparison/{roc,pr,model_correlation,varimp_heatmap,pareto_front}`.
  - Also logged as an artifact bundle `h2o-comparison-<run_id>`.
- Action: After reruns, verify the images exist under `<run_dir>/figures/` and appear in W&B. If still missing, share the run URLs; we’ll trace conditional paths.

## 7) `experiment_logs/` folder at repo root
- Not created by this codebase or Makefile targets. It appears to be a user‑side log sink (tee/redirect) that captured the CLI commands and outputs.
- The project’s own logs go under each `<run_dir>/h2o_logs/` and the Python logs to console.

## 8) W&B shows 4 runs, `local_runs` has only 3
- Root cause: run directory name collision — two runs started within the same second and shared `run_id`.
- Fix: Training pipeline now guarantees a unique `run_id`/`run_dir` by appending a numeric suffix when a directory already exists.
- Code: `src/training/pipeline.py` (post‑resolution of `run_dir`). Applies to all future runs.

## Next steps
- Re‑run the 1k suite with the new configs (no runtime cap) and confirm:
  - 4 distinct run folders under `local_runs/.../h2o/`.
  - Leaderboard/Pareto figures visible in W&B and present in the run folder.
  - New “best per category” figure generated.
- If you want the same changes for 10k/100k, I can apply them and queue runs.

