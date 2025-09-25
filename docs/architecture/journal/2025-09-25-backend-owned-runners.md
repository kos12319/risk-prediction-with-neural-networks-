# Backend-owned runners for orchestration

- Date: 2025-09-25
- Status: landed
- Tags: backends, orchestration, decoupling

## Summary
Introduce backend-owned runner modules for PyTorch and H2O that serve as the public orchestration entrypoints. This begins shifting ownership away from the shared base pipeline without changing external behavior.

## Impact
- configs: no changes required (CLIs and Make targets stay the same)
- Make: no changes to targets (`make train`, `make automl-h2o`, `make dryrun*` unchanged)
- code:
  - src/training/backends/pytorch/runner.py (new)
  - src/training/backends/h2o/runner.py (new)
  - src/training/backends/pytorch/__init__.py (route export via runner)
  - src/training/backends/h2o/__init__.py (route export via runner)

## Next
- Migrate more orchestration out of `base_pipeline.py` into backend-owned modules (dataset prep hooks, eval emission wiring) while keeping shared utilities for data/eval logic.
- Update developer docs with a short “new backend” skeleton once responsibility boundaries stabilize.

