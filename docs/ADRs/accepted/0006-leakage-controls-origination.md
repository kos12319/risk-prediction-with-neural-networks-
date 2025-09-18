# ADR 0006 — Leakage Controls at Origination

- Status: Accepted
- Date: 2025-09-18

## Context
Post‑origination columns (e.g., payments, recoveries) leak future information.

## Decision
Drop post‑origination/leaky columns by default; keep a provider‑agnostic baseline feature set that excludes pricing/scoring fields.

## Rationale
- Prevents leakage and optimistic bias.
- Establishes a fair baseline that generalizes across providers.

## Consequences
- Two config modes: provider‑agnostic (default) and provider‑aware (explicit).

## Alternatives Considered
- Include all columns and rely on training: rejected due to leakage.

