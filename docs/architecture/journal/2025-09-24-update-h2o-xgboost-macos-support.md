# Update README: H2O XGBoost on Apple Silicon supported

- Date: 2025-09-24
- Status: completed

## Summary
We removed an outdated note that implied H2O’s XGBoost does not work on macOS/M1. On current H2O (3.46.x), XGBoost is available on Apple Silicon when OpenMP (`libomp`) is installed. README now reflects the current state and documents how to force-disable XGBoost via config.

## Impact
- docs: README H2O AutoML section updated to state XGBoost support on macOS/Apple Silicon; added guidance to set `automl.exclude_algos: ['XGBoost']` and to install `libomp` if needed.
- configs: none (defaults unchanged).
- Make: none.
- code: none.

## Next
- Consider adding a convenience preset `configs/h2o/automl_no_xgb.yaml` that sets `automl.exclude_algos: ['XGBoost']` for users who want to avoid XGBoost.
