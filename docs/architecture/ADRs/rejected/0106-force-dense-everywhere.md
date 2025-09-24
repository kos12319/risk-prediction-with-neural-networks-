# ADR 0106 — Force Dense Representations End‑to‑End

- Status: Rejected
- Date: 2025-09-18

## Context
Categorical one‑hot features can grow design matrices large; staying sparse longer is efficient.

## Decision
Keep preprocessing outputs sparse; convert to dense with `.toarray()` only where required by the NN input.

## Rationale
- Balances memory/performance with model requirements.

## Alternatives Considered
- Force dense across the pipeline: simpler but wasteful and prone to memory issues.

