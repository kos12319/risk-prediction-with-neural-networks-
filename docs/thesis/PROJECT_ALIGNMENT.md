# Thesis Alignment Initiatives

This memo captures engineering-side proposals that keep the codebase in lockstep with the thesis scope defined in `THESIS_PROPOSAL.md`. Each item references the relevant research question or ADR so implementation can be tracked alongside the academic narrative.

## Time-Split Guardrails
- Enforce `split.method: time` as the resolved default for thesis runs, adding config validation and pytest checks to flag accidental stratified splits.
- Surface the behavior in documentation next to ADR 0001 (`docs/architecture/ADRs/accepted/0001-time-based-split.md`).

## Temporal Stability Selection
- Extend `src/cli/select.py` with forward/expanding-window CV so MI/L1 rankings aggregate across folds.
- Publish selection-frequency tables and stability metrics to support **H2** in `THESIS_PROPOSAL.md`.

## Calibration Toolkit
- Implement Platt, isotonic, and temperature scaling trained on validation predictions only.
- Report Brier score, ECE/MCE, and calibration curves to ground **RQ3/H3** experiments.

## Utility-Aligned Thresholding
- Enhance the evaluator (`src/eval/binary.py`, `src/training/base_pipeline.py`) with cost-matrix inputs, partial ROC in business FPR bands, and validation → test transfer tables.
- Supports **RQ4/H4** regarding expected-utility thresholds.

## Gradient-Boosting Baselines
- Integrate LightGBM/XGBoost/CatBoost via the model registry and Make targets, enabling the neural-vs-boosting comparison in **RQ5/H5**.

## Experiment Ledger
- Implement the append-only run index proposed in ADR 0012 (`docs/architecture/ADRs/proposed/0012-run-ledger.md`) to centralize run metadata, thresholds, and metrics for thesis tables.

## Provider-Regime Automation
- Provide paired configs/scripts that batch-run provider-agnostic vs provider-aware setups so **RQ1/H1** comparisons remain reproducible.

## Thesis-Ready Figures
- Extend the artifact pipeline (`src/training/base_pipeline.py` with backend hooks under `src/training/backends/*/pipeline.py`) to export calibration curves, partial ROC plots, and threshold sweeps directly into `local_runs/` for inclusion in thesis chapters.

## Loss-Template Configs
- Supply curated BCE and focal-loss config variants with multi-seed Make helpers, streamlining the loss-function and calibration study plan.

## Transformer-Assisted Selection (Exploratory)
- Prototype an attention-guided feature selector that complements MI/L1 rankings, addressing the open gap documented in `docs/thesis/from the previous reports lets explore one of the.md`.
