# Dropped Columns — Rationale and Watch‑outs

This note documents why certain raw columns are excluded from the default feature set, and flags additional columns to treat with caution. The goal is to avoid leakage, reduce instability and fairness risk, and keep features available at origination only.

## Summary of reasons
- Leakage: reveals post‑origination outcomes (payments, recoveries, last_* dates, hardship/settlement). These must be removed for fair evaluation.
- Provider‑aware or circular: variables set by the lender at underwriting (e.g., interest rate, grade) can encode the underwriting decision itself.
- High cardinality/noise: free text or highly granular IDs/location that balloon dimensionality or overfit (especially with one‑hot).
- Governance/fairness: attributes that are strong proxies for protected classes (e.g., fine‑grained geography).

## Columns intentionally dropped (non‑exhaustive)
- Provider/circular inputs
  - `int_rate`, `installment`, `funded_amnt`, `funded_amnt_inv`, `grade`, `sub_grade`
  - Rationale: set by the lender and entangled with the decision process; can create circular signal and unfair advantage vs baseline features.
- High‑cardinality or noisy categoricals
  - `emp_title` (free text; 5k+ unique in small samples)
  - `zip_code` (ZIP3/ZIP5; 700–>1000+ unique buckets)
  - Rationale: one‑hot expansion → large sparse design; prone to memorization; distribution drift over time. For `zip_code` specifically, also fairness risk (geography as a proxy).
- Identifiers and URLs
  - `id`, `member_id`, `url`
  - Rationale: identifiers or derived strings; not causal features; can leak structure of data collection.
- Post‑origination/leaky (examples; see config for full list)
  - `out_prncp`, `total_pymnt`, `last_pymnt_d`, `collection_recovery_fee`, `recoveries`, `last_credit_pull_d`
  - `hardship_*`, `debt_settlement_*`, `payment_plan_start_date`, `orig_projected_additional_accrued_interest`

The default config (`configs/pytorch_default.yaml`) codifies these via `data.features` (whitelist of allowed columns) and `data.leakage_cols` (blacklist for safety).

## Alternatives if you want the signal
- `emp_title`
  - Normalize to job families: cluster titles locally (e.g., `skrub` SimilarityEncoder or TF–IDF + KMeans) or map to SOC/Census/ESCO codes, then encode the small set of families.
  - Keep train‑only fitting and persist mappings to avoid drift; check for bias.
- `zip_code`
  - Prefer coarser `addr_state` (already used). If granular location is needed, aggregate to stable regions or join leakage‑safe macro indicators by date (e.g., unemployment), not by fine geography.
- `grade` / `sub_grade` / `int_rate`
  - If running “provider‑aware” experiments, enable these explicitly in a separate config and label results appropriately. Keep a provider‑agnostic baseline for fair comparisons.

## Other columns to review (potentially leaky or unstable)
- User‑entered text: `title` (noisy, overlaps with `purpose`), `desc` (often missing)
- Listing flags: `initial_list_status`, `pymnt_plan` (can correlate with post‑listing processes)
- Policy/administrative: `policy_code`, rarely informative and may shift over time

When in doubt: ensure a column is available at origination, stable across vintages, and not a proxy for decisions or protected attributes. If a column survives that filter but is high‑cardinality, prefer target encoding with strict train‑only/OOF safeguards or hashed encoding.
