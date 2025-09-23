# Neural Network Capacity Sweep Proposal

## Context
- Dataset: `thesis_data_full.csv` (LendingClub accepted loans 2007-2018) filtered to final outcomes; positive class remains `Charged Off = 0`.
- Splitting: time-based on `issue_d`; validation carved from the training period before any oversampling; oversampling applies to the training subset only.
- Thresholding: select the decision threshold on the validation slice using the configured strategy (default: Youden J), then apply it unchanged to the test split.
- Pipeline: execute runs through `make train` (or `make cpu-train`) with updated configs; keep leakage controls and engineered features unchanged.
- Baseline reference: current provider-agnostic MLP with layers `[256, 128, 64, 32]`, dropout `[0.4, 0.3, 0.2, 0.2]`, batch norm enabled, ~30 epochs with early stopping.

## Goals
- Assess whether scaling width and depth improves discrimination (ROC AUC, average precision) on the validation and held-out test periods without violating reproducibility or leakage guardrails.
- Measure trade-offs in training time, convergence stability, and class-0 recall at the validation-selected threshold.

## Proposed Architectures
Run each configuration with the full feature set from `configs/default.yaml`, copying the config per experiment and adjusting only the `model.layers` and `model.dropout` entries. Keep optimizer, batch size, seeding, and early stopping unchanged.

1. **Compact baseline check**
   - Layers: `[128, 64, 32]`
   - Dropout: `[0.2, 0.1, 0.1]`
   - Purpose: confirm that the existing performance gap is not due to over-capacity; expect lower AUC if capacity is truly helpful.

2. **Deeper same-width stack**
   - Layers: `[256, 256, 128, 64, 32]`
   - Dropout: `[0.4, 0.3, 0.25, 0.2, 0.1]`
   - Purpose: test whether an extra hidden block captures higher-order interactions while keeping the intake width unchanged.

3. **Wider front end**
   - Layers: `[512, 256, 128, 64]`
   - Dropout: `[0.4, 0.3, 0.2, 0.1]`
   - Purpose: exploit the large sample size with a wider first layer; monitor for overfitting via validation loss and early stopping.

4. **Aggressive width sweep**
   - Layers: `[768, 384, 192, 96]`
   - Dropout: `[0.5, 0.35, 0.25, 0.15]`
   - Purpose: stress-test capacity limits. Use only if hardware resources comfortably handle the larger parameter count; be ready to raise dropout further if validation loss diverges.

## Evaluation Checklist
- Use `make train CONFIG=...` for each variant and capture run IDs in `local_runs/`.
- Record validation ROC AUC, test ROC AUC, average precision, threshold, and confusion metrics (with `pos_label=0`).
- Compare each variant to the existing baseline; promote any winning architecture to `configs/default.yaml` only after confirming a consistent lift across multiple seeds or splits.
- If significantly better capacity emerges, repeat the winning configuration on the provider-aware feature set to evaluate portability vs accuracy impacts.

## Logging & Next Steps
- Store metrics in the corresponding run folders and note outcomes in `docs/experiments/run/` once executed; move unsuccessful trials to `docs/experiments/rejected/` with a short rationale.
- Update this proposal as needed if new constraints (e.g., resource limits, feature set changes) arise.

## Dropout Adjustment Guidance
- Compact baseline: keep dropout light at `[0.2, 0.1, 0.1]` so reduced capacity still learns; increase by 0.05-0.1 only if validation AUC plateaus well below baseline.
- Deeper same-width stack: hold early layers at `0.4` and `0.3`, then taper to `0.25`, `0.2`, and `0.1`; tighten the tail (e.g., drop last two layers by 0.05) if underfitting shows up.
- Wider front end: start at `[0.4, 0.3, 0.2, 0.1]`; bump the first layer to `0.5` if validation loss diverges, or shave each rate by 0.05 when learning stalls.
- Aggressive width sweep: use `[0.5, 0.35, 0.25, 0.15]`; for overfitting raise the first rate to `0.55`, for underfitting lower the entire vector by ~0.05 while keeping the descending pattern.
