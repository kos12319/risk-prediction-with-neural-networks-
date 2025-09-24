# ADR 0103 — Include Pricing/Scoring Fields in Provider‑Agnostic Model

- Status: Rejected
- Date: 2025-09-18

## Context
Interest rate, grade, sub_grade, and installment reflect provider decisions and pricing signals.

## Decision
Exclude these fields from the provider‑agnostic baseline; allow a separate provider‑aware config to include them explicitly.

## Rationale
- Avoids provider leakage; keeps baseline comparable across providers and time.

## Alternatives Considered
- Always include: higher AUC but less generalizable and potentially leaky.

