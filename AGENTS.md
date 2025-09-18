# AGENTS — Working Guide (Read README First)

Start here: open README.md and follow Quick Start and Makefile Targets. Treat README as the authoritative source for setup, CLI usage, and Makefile commands.

This guide adds repo‑specific guardrails and conventions that are easy to miss. Avoid hardcoding paths; everything is config‑driven.

## Makefile‑First Policy
- Always run workflows via the Makefile. Do not call `python -m src...` directly in routine use.
- Prefer these targets for all operations:
  - Training: `make train CONFIG=... [NOTES=...] [PULL=true]` or CPU: `make cpu-train CONFIG=...`
  - Dry run: `make dryrun CONFIG=...`
  - Feature selection: `make select CONFIG=... METHOD=mi|l1`
  - Column dictionary: `make dict CONFIG=... [CSV=path]`
  - W&B: `make wandb-login`, `make pull-run RUN=...`, `make pull-all [ENTITY=...] [PROJECT=...]`
  - Deps: `make deps-tools`, `make deps-compile`, `make deps-sync`
  - Cleanup: `make clean-local-runs`, `make clean-wandb-local`, `make clean-local-history`, `make clean-all-local`, `make clean-venv`
- If you need a new operation, add a Makefile target rather than introducing bespoke shell commands in docs or scripts.
- Pass configuration via Makefile variables (not hardcoded flags): `CONFIG`, `NOTES`, `PULL`, `METHOD`, `CSV`, `RUN`, `FORCE`, `ENTITY`, `PROJECT`.
- Rationale: Make targets enforce safe environment settings (thread limits, headless plotting) and keep runs reproducible.

## Evaluation Invariants (don’t break)
- Use time‑based split by `issue_d` for test; older → train, newer → test.
- Hold out validation from the training period; oversample the training subset only.
- Choose threshold on validation using the configured strategy (`fixed|youden_j|f1`); report test metrics at that fixed threshold.
- Respect `eval.pos_label` (default: 0 = Charged Off). Curves/metrics must reflect the configured positive class.
- Seed Python, NumPy, PyTorch, and DataLoader workers for reproducibility.

## Data Handling & LFS
- Don’t commit large uncompressed data outside `data/raw/archives/` (LFS) or `data/raw/full/` (gitignored).
- Archives are LFS‑tracked (`*.zip`, `data/raw/archives/**`). If you see LFS pointers, run `git lfs pull` then unzip into the appropriate folder.

## Coding Style & Conventions
- Python 3.10+; type hints required for public functions.
- Naming: snake_case (functions/variables), PascalCase (classes), UPPER_SNAKE (constants).
- Keep modules cohesive and small; prefer pure functions over side effects.
- Design to avoid leakage: time‑split, train‑only oversampling, drop post‑origination features (`data.drop_leakage`).

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
- Selection CLI: ensure it resolves `extends` like training does.
- Headless plotting: use `MPLBACKEND=Agg`; set `XDG_CACHE_HOME=.cache` and `MPLCONFIGDIR=.mplcache` if needed; limit BLAS threads if OMP errors appear.

## If You’re Lost
- Read README.md (Quick Start, Makefile Targets).
- If data is missing, `git lfs pull`, then update `data.csv_path` to a sample CSV under `data/raw/samples/`.
- See `docs/ADRs/` (time split rationale and proposals) and `docs/PAIN_POINTS.md`.
- Ask for clarification before changing evaluation protocols or data handling.
