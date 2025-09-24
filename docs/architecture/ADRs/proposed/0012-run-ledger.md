# ADR 0012 — Run Ledger for Reproducibility

- Status: Proposed
- Date: 2025-09-18

## Context
Comparing runs across time is cumbersome without a consolidated index.

## Proposal
Maintain an append‑only CSV ledger (e.g., `reports/experiments.csv`) with key fields (run_id, config, seed, class prevalence, ROC AUC/AP, threshold, pos_label, duration). Provide a small CLI to rebuild or append entries.

## Rationale
- Simplifies thesis tables and experiment comparisons.
- Makes provenance explicit.

## Consequences
- Slight reporting overhead; one more artifact to maintain.

## Alternatives Considered
- Ad‑hoc comparisons in notebooks: error‑prone, less reproducible.

