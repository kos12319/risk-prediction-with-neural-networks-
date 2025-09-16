# Thesis Pivot Plan — Experiment Tracking and Rationale

This document proposes concrete changes to make the MSc CS thesis scope explicit and to improve experiment tracking, reproducibility, and rationale capture.

## Objectives
- Make the thesis context unmistakable (scope, goals, evaluation protocol).
- Systematically record experiments, environment, and decisions.
- Reduce friction when writing the thesis chapters and reproducing figures/tables.

## Make Thesis Context Obvious
- README: add a short “Thesis” section (degree, university, year, topic, objectives, dataset provenance).
- New `docs/thesis/OVERVIEW.md`: problem statement, hypotheses, datasets, evaluation protocol, risks.
- Add `CITATION.cff` and a “How to cite” block in README.
- Add a short “Data Ethics” note: no PII, licensing, and usage constraints.

## Experiment Ledger (Cross‑Run Index)
- File: `reports/experiments.csv` (append‑only). Columns:
  - `run_id,timestamp,git_sha,config_path,backend,seed,n_train,n_test,class_prevalence,roc_auc,ap,threshold,pos_label,duration_sec,notes`
- Script: `src/cli/experiments_index.py` to (re)build the ledger from `reports/runs/*/` and to append after each run.
- Usage: cite this ledger in thesis tables (ablation summaries, method comparisons).

## Per‑Run Artifact Upgrades
- Provenance files under each `reports/runs/<run_id>/`:
  - `git.json` (commit SHA, branch, dirty flag);
  - `env.txt` (`pip freeze`), `python_version.txt`, `platform.txt`;
  - `data_manifest.json` (dataset path, size, mtime, sha256, train/test date ranges, class prevalence);
  - `seeds.json` (numpy, torch, python `random`).
- Statistical context:
  - `metrics_ci.json` (bootstrap CIs for ROC AUC/AP);
  - `calibration.png` + `brier_score.json`;
  - `cohort_metrics.csv` (metrics by `issue_d` month/quarter to show drift).
- Run README template: add “Threats to validity” and “Related ADRs” sections.

## Decision & Rationale Tracking (ADRs)
- Keep ADRs under `docs/adr/`; add `docs/adr/TEMPLATE.md` (Context, Decision, Rationale, Consequences, Alternatives, Status).
- Require ADRs for: time split (added), class weighting vs oversampling, threshold selection policy, backend choice, feature subset method.
- Reference relevant ADR IDs in each run README.

## Configuration & Sweeps
- Create `experiments/*.yaml` to define experiment matrices (list of configs/overrides).
- New CLI `src/cli/sweep.py` to run matrices with deterministic seeds (`base_seed + idx`) and write ledger rows.
- Run naming: `run_YYYYMMDD_HHMMSS__cfg-<name>__pytorch__pos<0|1>__seed<NN>`.

## Evaluation Hygiene
- Threshold selection on validation:
  - Choose operating threshold on validation set (not test); report test metrics at that fixed threshold.
  - Record both validation‑selected threshold and final test performance.
- Prefer class weighting or focal loss over oversampling; if using ROS, apply only to the train subset (never validation/test), cap the effective ratio, and seed.
- Optional: add forward‑chaining temporal CV for tuning; keep a final untouched out‑of‑time holdout for reporting.

## Reproducibility & Testing
- Determinism: add a `set_seed()` utility; seed numpy, python `random`, torch (incl. dataloaders). Store seeds in each run.
- Tests (`pytest`):
  - Time split monotonicity and non‑overlap;
  - Preprocessing determinism and stable feature count;
  - Model I/O roundtrip on a toy sample.
- Script `scripts/freeze_env.sh` to capture env metadata into the current run folder.

## Roadmap (Incremental)
1) Add README thesis section and `docs/thesis/OVERVIEW.md`.
2) Add ledger scaffolding and one‑line append after training completes.
3) Capture provenance (git/env/data/seeds) in run folders.
4) Move threshold selection to validation; implement split‑before‑oversample.
5) Add ADR template and backfill a few key ADRs.
6) Optional: sweeps CLI + temporal CV support.

## Acceptance Criteria
- Every run produces a complete, self‑contained bundle for thesis figures/tables.
- A single CSV (experiments ledger) summarizes key metrics across runs.
- Major decisions are documented as ADRs, referenced from runs.

## Ownership / Next Actions
- Confirm whether to standardize on PyTorch (recommended) and remove TF code.
- Approve the ledger schema and run folder additions.
- I can implement steps (1)–(4) next if you agree.

