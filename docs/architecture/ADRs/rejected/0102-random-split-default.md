# ADR 0102 — Random Split as Default

- Status: Rejected
- Date: 2025-09-18

## Context
Random train/test splits for temporal problems overstate performance by mixing cohorts.

## Decision
Do not use random split as default; default to time‑based split by `issue_d`.

## Rationale
- Better reflects deployment scenario and data drift.

## Alternatives Considered
- Random split for quick debugging: acceptable if documented and not used for reporting.

