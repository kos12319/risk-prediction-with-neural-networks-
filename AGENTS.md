# AGENTS — Working Guide (Read README First)

Start here: open README.md and READ it. READ the Makefile help and the Makefile Targets if you want to run an action. Treat README as the authoritative source for setup and repo information index. Read the Makefile for CLI usage, and Makefile commands.

This guide adds repo‑specific guardrails and conventions that are easy to miss. Avoid hardcoding paths; everything is config‑driven.

## Maintenance Policy
- Backward compatibility is not required; prefer simplifying the codebase over preserving deprecated behavior.
- Remove dead code paths and leftover files as part of routine changes so agents and contributors inherit a clean tree.

## Makefile‑First Policy
- Always run workflows via the Makefile. Do not call `python -m src...` directly in routine use.
- Discover available operations by reading the `Makefile` (and `make help` if present). Do not rely on copied command snippets here.
- `make train` runs the PyTorch backend; use `make automl-h2o` for H2O AutoML experiments. When using the H2O CLI/Make target, `model.backend` is optional.
- Use `make dryrun` for PyTorch smoke tests and `make dryrun-h2o` for H2O smoke tests; both commands clean up temporary artifacts automatically. For temporal CV smoke tests, use `make dryrun-cv` (PyTorch) or `make dryrun-h2o-cv` (H2O; requires Java). For a tiny end-to-end CV followed by final training, use `make cv-train` (PyTorch, smoke config).
- Catalog helpers: `make run-catalog` builds `local_runs/_catalog.json`; `make run-catalog-report` renders `local_runs/index.md` for quick browsing.
- If you need a new operation, add a Makefile target rather than introducing bespoke shell commands in docs or scripts.
- Pass configuration via Makefile variables (see the `Makefile` for supported variables and defaults). Avoid hardcoded flags in ad-hoc commands.
- Rationale: Make targets enforce safe environment settings (thread limits, headless plotting) and keep runs reproducible.
- Config layout: PyTorch presets extend files under `configs/pytorch/` (e.g., `configs/pytorch/base.yaml`), H2O AutoML presets extend `configs/h2o/`, and the example third backend lives under `configs/template/`.

## Evaluation Invariants (don’t break)
- Use time‑based split by `issue_d` for test; older → train, newer → test.
- Hold out validation from the training period; oversample the training subset only.
- Choose threshold on validation using the configured strategy (`fixed|youden_j|f1`); report test metrics at that fixed threshold.
- Respect `eval.pos_label` (default: 0 = Charged Off). Curves/metrics must reflect the configured positive class.
- Seed Python, NumPy, PyTorch, and DataLoader workers for reproducibility.
- Temporal k-fold CV is available via `split.cv` (expanding window). Aggregated metrics live in `reports/cv_metrics.json`; per-fold artifacts write to `run_dir/folds/fold_XX/`. Set `train_full_after: true` to refit on the full dataset after CV.

## Dataset Context (LendingClub)
- Dataset: LendingClub consumer installment loans, 2007–2018 vintages.
- Files: “accepted” (funded, final statuses) and “rejected” (declined, limited covariates).
- Labels: derived from funding outcomes (e.g., Charged Off vs Fully Paid); beware right‑censoring in recent vintages.
- Leakage policy: features must be available at origination only; drop post‑event fields (payments, recoveries, last_* dates, hardship/settlement) consistently.
- Positive class convention: default is `eval.pos_label=0` (Charged Off); ensure curves/metrics/thresholding use the configured `pos_label`.
- Splits: time‑based by `issue_d` (older→train, newer→test); carve validation from the training period only.

## Data Handling & LFS
- Don’t commit large uncompressed data outside `data/raw/archives/` (LFS) or `data/raw/full/` (gitignored).
- Archives are LFS‑tracked (`*.zip`, `data/raw/archives/**`). If you see LFS pointers, run `git lfs pull` then unzip into the appropriate folder.

## Coding Style & Conventions
- Python 3.10+; type hints required for public functions.
- Naming: snake_case (functions/variables), PascalCase (classes), UPPER_SNAKE (constants).
- Keep modules cohesive and small; prefer pure functions over side effects.
- Design to avoid leakage: time-split, train-only oversampling, drop post-origination features (`data.drop_leakage`).
- Shared pipeline utilities live in `src/training/base_pipeline.py`.
- Backends must import the stable interfaces from `src/training/interfaces.py` (`BackendPipeline`, `DatasetBundle`, `BackendTrainingResult`, `RunContext`) rather than importing these directly from `base_pipeline`.
- Backend-specific orchestration sits in `src/training/backends/{pytorch,h2o,template}/pipeline.py` and owns backend concerns (naming, tags, extra artifacts). The shared utilities must not branch on backend types.

## Testing
- Use pytest; place tests under `tests/` as `test_*.py`.
- Suggested tests: preprocessing invariants, time‑split monotonicity, model I/O round‑trip, thresholding correctness, run artifacts schema.

## PR Hygiene
- Commits: concise, imperative subject; group related changes.
  - Examples: `feat(training): add focal loss option`, `fix(data): compute credit_history_length vs issue_d`.
- PRs: include what/why, config used, before/after ROC AUC (from run `metrics.json`), and figures under `reports/figures/`.

## Known TODOs / Watch‑outs
- Oversampling isolation: carve validation from train before oversampling; oversample train subset only.
- Determinism: seed Python/NumPy/Torch; use a seeded generator for Torch splits.
- Threshold selection: compute on validation; apply fixed threshold to test.
- Dense conversion: prefer `.toarray()` over `.todense()` where applicable.
- Feature name mapping after OHE: avoid brittle string splits; use encoder introspection.
- Selection CLI: now resolves `extends` via the shared loader; keep `tests/test_training_config.py` updated when adding configs.
- H2O backend requires a working Java runtime; the CLI fails fast if `java -version` is blocked (e.g., sandboxed environments).
- Headless plotting: use `MPLBACKEND=Agg`; set `XDG_CACHE_HOME=.cache` and `MPLCONFIGDIR=.mplcache` if needed; limit BLAS threads if OMP errors appear.
- Medium backlog focus: config validation guardrails and temporal CV are implemented; a first version of the run catalog manifest is available via `make run-catalog` (see `docs/architecture/PAIN_POINTS.md`).

## If You’re Lost
- Read README.md (Quick Start, Makefile Targets).
- If data is missing, `git lfs pull`, then update `data.csv_path` to a sample CSV under `data/raw/samples/`.
- See `docs/architecture/ADRs/` (time split rationale and proposals) and `docs/architecture/PAIN_POINTS.md`.
- Ask for clarification before changing evaluation protocols or data handling.

## Automation Scripts
- `scripts/run_codex_yolo.py` — repeatedly launches the Codex CLI in non-interactive mode (`codex exec --dangerously-bypass-approvals-and-sandbox`) with the automation prompt; stops after two hours or on failure.
- `scripts/codex_runner_daemon.py` — daemon supervisor that spawns the runner script and checks every five minutes that runtime stays within the two-hour ceiling.
- Environment knobs:
  - `CODEX_CLI_CMD` overrides the Codex executable/arguments (default: `codex exec --dangerously-bypass-approvals-and-sandbox`).
  - `CODEX_CLI_SLEEP` waits between successful iterations (seconds).
  - `CODEX_CLI_FAIL_SLEEP` backs off after failed launches (default 5s).
  - `CODEX_CLI_MAX_FAILURES` stops the runner after N consecutive failures (default 100).
  - `CODEX_CLI_CWD` forces the working directory for Codex runs.
