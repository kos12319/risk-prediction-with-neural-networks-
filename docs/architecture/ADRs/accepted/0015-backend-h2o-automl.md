# ADR 0015 — H2O AutoML as a Comparative Backend

- Status: Accepted
- Date: 2025-09-24

## Context
Tree‑based AutoML provides strong baselines and explainability tools (varimp, SHAP) that complement neural models.

## Decision
Support H2O AutoML as an alternate backend for training/evaluation. Keep PyTorch as the primary backend; use H2O for benchmarking, feature diagnostics, and subsets.

## Rationale
- Competitive baselines and interpretability to guide feature work.
- Cross‑validation of evaluation invariants across backends.

## Consequences
- Separate CLI/Make target to avoid coupling neural and H2O runtimes.
- Additional dependency footprint (documented and pinned).

## Alternatives Considered
- Only neural: limits comparative insight.
- Scikit‑learn only: fewer AutoML/explainability features compared to H2O.

## Implementation Notes
- Makefile: `automl-h2o` target.
- Configs: `configs/h2o/*.yaml` presets.
- Code: `src/cli/automl_h2o.py`, `src/training/train_h2o.py`; shares preprocessing/splits/eval path.

