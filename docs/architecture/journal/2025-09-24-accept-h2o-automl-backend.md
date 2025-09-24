# H2O AutoML accepted as comparative backend

- Date: 2025-09-24
- Status: landed
- Tags: backend, automl, baselines

## Summary
Added H2O AutoML as a comparative backend to provide strong tree-based baselines and interpretability (varimp, SHAP) alongside the PyTorch MLP.

## ADRs
- 0015 — see docs/architecture/ADRs/accepted/0015-backend-h2o-automl.md

## Impact
- Make: `automl-h2o` target runs AutoML with presets under `configs/h2o/`.
- Code: `src/cli/automl_h2o.py`, `src/training/train_h2o.py` sharing preprocessing/eval.
- Docs: experiment reports under `docs/experiments/run/` include H2O results.

## Next
- Use H2O varimp/SHAP to guide neural feature engineering and subset decisions.

