# ADR 0007 — Single Run Folder Under local_runs/

- Status: Accepted
- Date: 2025-09-18

## Context
Scattering artifacts across multiple top‑level folders complicates tracking and cleanup.

## Decision
Write all artifacts for a run into a single folder `local_runs/run_YYYYMMDD_HHMMSS/` with relative paths referenced in README.

## Rationale
- Self‑contained run bundles are easier to inspect, copy, and archive.
- Simplifies automation and W&B artifact logging.

## Consequences
- Deprecated `reports/` and `models/` as global sinks.

## Alternatives Considered
- Multiple top‑level artifact folders: rejected due to fragmentation.

