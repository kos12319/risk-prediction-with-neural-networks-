# Optional W&B tracking backend enabled

- Date: 2025-09-24
- Status: landed
- Tags: tracking, wandb, reproducibility

## Summary
Integrated Weights & Biases (W&B) as an optional experiment tracking backend with helpers for login and pulling runs. Local/offline runs remain unchanged when disabled.

## ADRs
- 0014 — see docs/architecture/ADRs/accepted/0014-tracking-backend-wandb.md

## Impact
- Config: `tracking.backend: wandb` or `tracking.wandb.enabled: true` to enable.
- Make: `wandb-login`, `pull-run`, `pull-all`, `clean-cloud-history`.
- Code: `src/training/pipeline.py` conditional init and artifact logging; `src/cli/wandb_*` helpers.

## Next
- Consider toggling selection CLI to optionally log to W&B in the future.

