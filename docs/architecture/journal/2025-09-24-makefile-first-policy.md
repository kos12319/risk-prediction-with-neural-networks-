# Makefile-first workflow adopted

- Date: 2025-09-24
- Status: landed
- Tags: docs, dx, reproducibility

## Summary
Adopted a Makefile-first policy for all routine workflows (training, AutoML, selection, exploration, docs). Targets encode safe env defaults and parameterization to improve reproducibility and developer experience.

## ADRs
- 0013 — see docs/architecture/ADRs/accepted/0013-makefile-first-policy.md

## Impact
- Makefile: targets `train`, `automl-h2o`, `cpu-train`, `select`, `dict`, `explore`, `dryrun`, `wandb-*`, `docs-*`.
- Docs: README and guides updated to reference Make targets.
- Env: thread limits and headless plotting applied consistently.

## Next
- Add any new operations as Make targets rather than bespoke commands.

