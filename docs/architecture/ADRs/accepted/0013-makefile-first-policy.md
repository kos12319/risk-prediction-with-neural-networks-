# ADR 0013 — Makefile‑First Workflow and Safe Defaults

- Status: Accepted
- Date: 2025-09-24

## Context
Directly invoking Python modules with ad‑hoc flags leads to inconsistent environments (BLAS threads, plotting backends), brittle command usage, and poor reproducibility.

## Decision
Adopt a Makefile‑first policy. All routine workflows (training, AutoML, selection, exploration, docs) run via Make targets which encode safe environment settings and parameterization via variables.

## Rationale
- Reproducibility: centralizes commands and env vars.
- Safety: sane thread limits and headless plotting reduce flakiness.
- Discoverability: `make help` documents available operations.

## Consequences
- Documentation and examples reference Make targets, not raw `python -m …`.
- New operations are added as Make targets.

## Alternatives Considered
- Script‑per‑task shell wrappers: more maintenance; less transparent.

## Implementation Notes
- Makefile targets: `train`, `automl-h2o`, `cpu-train`, `select`, `dict`, `explore`, `dryrun`, W&B helpers, docs helpers.
- Safe env: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 … MPLBACKEND=Agg` baked into targets.
- Config via vars: `CONFIG=…`, `AUTOML_CONFIG=…`, `METHOD=…`, `NOTES=…`, `PULL=true`.

