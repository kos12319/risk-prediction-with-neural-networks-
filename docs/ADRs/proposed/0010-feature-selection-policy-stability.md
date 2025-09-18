# ADR 0010 — Feature Selection Policy with Stability Criterion

- Status: Proposed
- Date: 2025-09-18

## Context
Single‑method, single‑split rankings (MI or L1) can be unstable across time.

## Proposal
Combine MI and L1 rankings (e.g., normalized scores + average rank) and require stability across temporal CV folds (e.g., selection frequency) before adopting a subset.

## Rationale
- Reduces sensitivity to a single method and time window.
- Encourages subsets that generalize.

## Consequences
- Slightly more compute; more transparent selection rationale.

## Alternatives Considered
- Pick one method and one split: faster but higher variance.

