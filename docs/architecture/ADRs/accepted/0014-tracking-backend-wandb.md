# ADR 0014 — Optional Experiment Tracking via Weights & Biases (W&B)

- Status: Accepted
- Date: 2025-09-24

## Context
Experiment tracking and artifact management benefit from an external system for dashboards and provenance, but local/offline runs must remain possible.

## Decision
Support W&B as an optional tracking backend. Enable via config (`tracking.backend: wandb` or `tracking.wandb.enabled: true`). Default to local‑only when disabled.

## Rationale
- Better visibility into metrics and model comparisons.
- Seamless artifact logging when enabled; zero coupling when disabled.

## Consequences
- Requires environment variables (`WANDB_API_KEY`, `WANDB_ENTITY`) to upload.
- Make targets provide helpers to login and pull/sync run artifacts.

## Alternatives Considered
- Always‑on tracking: too heavy for local iteration; rejected.
- Ad‑hoc CSV/plots only: less discoverable and harder to compare.

## Implementation Notes
- Config: `tracking.backend`, `tracking.wandb.*` keys with templates for names/groups/tags.
- Code: `src/training/pipeline.py` conditional W&B init and artifact logging; helpers under `src/cli/wandb_*`.
- Makefile: `wandb-login`, `pull-run`, `pull-all`, `clean-cloud-history`.

