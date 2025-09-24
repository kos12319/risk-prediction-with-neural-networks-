# Standardized positive class to 0 (Charged Off)

- Date: 2025-09-24
- Status: landed
- Tags: eval, metrics

## Summary
Standardized the default positive class to 0 (Charged Off) across the pipeline. Metrics, curves, threshold selection, and probability alignment now consistently respect `eval.pos_label`.

## ADRs
- 0011 — see docs/architecture/ADRs/accepted/0011-positive-class-convention.md

## Impact
- Config: `eval.pos_label: 0` in `configs/default.yaml` and variants.
- Code: probability alignment and metric computation updated in `src/training/pipeline.py` and H2O path.
- Docs: README clarifies the convention and override.

## Next
- Ensure confusion matrix plots label the positive class explicitly in figures.

