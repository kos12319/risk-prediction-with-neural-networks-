# ADR 0016 — Provider‑Agnostic Feature Set by Default; Provider‑Aware Optional

- Status: Accepted
- Date: 2025-09-24

## Context
Pricing/scoring fields (e.g., `int_rate`, `grade`, `sub_grade`, `installment`, `funded_amnt`) can inflate apparent performance and reduce generality across providers.

## Decision
Default to a provider‑agnostic feature set that excludes pricing/scoring fields. Provide separate provider‑aware configs when such fields are intentionally included for within‑provider baselines.

## Rationale
- Promotes generalizable modeling across providers or eras.
- Prevents leakage‑like effects from pricing decisions.

## Consequences
- Slightly lower baseline metrics vs provider‑aware models, but more robust comparisons.
- Clear separation of experiment goals (agnostic vs aware) in configs and reports.

## Alternatives Considered
- Always include pricing fields: inflated performance; harder to compare across regimes.

## Implementation Notes
- Configs: `configs/default.yaml` (agnostic), `configs/pytorch/provider_aware.yaml` (aware).
- Docs: README calls out the split; experiments label runs accordingly.

