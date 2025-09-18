# AGENTS — Working Guide (Read README First)

Start here: open README.md and follow Quick Start and Makefile Targets. Treat README as the authoritative source for setup, CLI usage, and Makefile commands.

This guide adds repo‑specific guardrails and conventions that are easy to miss. Avoid hardcoding paths; everything is config‑driven.

## Makefile‑First Policy
- Always run workflows via the Makefile. Do not call `python -m src...` directly in routine use.
- Discover available operations by reading the `Makefile` (and `make help` if present). Do not rely on copied command snippets here.
- If you need a new operation, add a Makefile target rather than introducing bespoke shell commands in docs or scripts.
- Pass configuration via Makefile variables (see the `Makefile` for supported variables and defaults). Avoid hardcoded flags in ad‑hoc commands.
- Rationale: Make targets enforce safe environment settings (thread limits, headless plotting) and keep runs reproducible.

## Evaluation Invariants (don’t break)
- Use time‑based split by `issue_d` for test; older → train, newer → test.
- Hold out validation from the training period; oversample the training subset only.
- Choose threshold on validation using the configured strategy (`fixed|youden_j|f1`); report test metrics at that fixed threshold.
- Respect `eval.pos_label` (default: 0 = Charged Off). Curves/metrics must reflect the configured positive class.
- Seed Python, NumPy, PyTorch, and DataLoader workers for reproducibility.

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
