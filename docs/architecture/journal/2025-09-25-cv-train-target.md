# Add PyTorch cv-train smoke target

- Date: 2025-09-25
- Status: done
- Tags: temporal-cv, makefile, orchestration

## Summary
Added a Make target `cv-train` that runs a tiny temporal CV (2 folds) and then fits a final model on the full dataset. This provides a Makefile-first entry point for full CV workflows while preserving backend separation.

## Impact
- Makefile: new target `cv-train`
- Configs: `configs/pytorch/cv_full_train_smoke.yaml` (2 folds, 2 epochs, train_full_after=true)
- README: documented the new target under Dry Run/Temporal CV
- Future Extensions: marked the CV orchestrator item as partially delivered for PyTorch

## Next
- Consider adding an H2O counterpart (`cv-automl-h2o`) once Java availability checks and runtime budgets are tuned for a smoke config.
