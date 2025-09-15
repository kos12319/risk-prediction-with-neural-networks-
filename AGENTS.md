# Repository Guidelines

## Project Structure & Module Organization
- `src/` Python package:
  - `data/` load, target mapping, time/random split
  - `features/` preprocessing (impute/scale/one‑hot)
  - `models/` neural network builder (Keras)
  - `training/` end‑to‑end train orchestration
  - `eval/` metrics and plots
  - `cli/` runnable entry points
- `configs/` YAML configs (provider‑agnostic/aware).
- `models/`, `reports/` artifacts; `docs/` notes; `notebooks/` for EDA.

## Build, Test, and Development Commands
- Create env and install deps:
  - `python -m venv .venv && . .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run training (config‑driven):
  - `python -m src.cli.train --config configs/default.yaml`
  - Example override: edit `data.csv_path` to your CSV.

## Coding Style & Naming Conventions
- Python 3.10+, type hints required for public functions.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Keep modules cohesive and small; prefer pure functions over side effects.
- Avoid data leakage: time‑based split by default; oversample only on the training split.

## Testing Guidelines
- Use `pytest` (not bundled). Place tests under `tests/` as `test_*.py`.
- Suggested targets: preprocessing invariants, split correctness, model I/O.
- Example: `pytest -q` (after `pip install pytest`). No minimum coverage enforced yet.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject; group related changes.
  - Examples: `feat(training): add focal loss option`, `fix(data): compute credit_history_length vs issue_d`.
- PRs must include:
  - What/why, configs used, and before/after metrics (ROC AUC) from `reports/metrics.json`.
  - Any figures (learning curves) under `reports/figures/`.

## Security & Configuration Tips
- Never commit secrets or PII. Prefer pointing `data.csv_path` to local files.
- Large raw datasets should live outside git; commit only configs and code. Model artifacts are acceptable if <100MB.

## Agent‑Specific Instructions
- Respect config‑driven design; don’t hardcode paths.
- Maintain leakage controls and time‑aware validation unless a task explicitly requests otherwise.

