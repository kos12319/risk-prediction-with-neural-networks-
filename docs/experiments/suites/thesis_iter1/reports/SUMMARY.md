# Thesis Iter 1 — Cross‑Dataset Summary

## Winners by Dataset
| Dataset | Winner Run | Features | Avg Precision | ROC AUC |
|---|---|---:|---:|---:|
| 1k   | run_20250925_020521 | 39 | 0.3148 | 0.7313 |
| 10k  | run_20250925_023120 | 43 | 0.4601 | 0.7591 |
| 100k | run_20250925_032002 | 43 | 0.4524 | 0.7435 |
| full | run_20250925_070714 | 43 | 0.3934 | 0.7093 |

## Key Findings
- Feature set: The enriched 43‑feature set including `int_rate`, `grade/sub_grade`, and `installment` wins on 10k/100k/full. On 1k, the leaner 39‑feature set without pricing/grade performs best, suggesting overfitting risk from high‑cardinality dummies at very small scale.
- Scale vs performance: AUCPR improves markedly from 1k → 10k, is comparable at 100k, and is lower on full — consistent with harder, more recent test periods and class imbalance dynamics.
- Top drivers: Across larger datasets, `int_rate`, term (36/60), grade bands, `dti`, and capacity proxies (`income_to_loan_ratio`, `mort_acc`, `annual_inc`) dominate variable importance.

## Comparative Figure
- AUCPR and ROC AUC of winners by dataset size: `docs/experiments/suites/thesis_iter1/reports/aupr_roc_winners_by_size.svg`

## Recommendations
- Default features: Use the 43‑feature set for production‑scale data; prefer the 39‑feature set for tiny samples (≤1k) to limit variance.
- Validation protocol: Continue time‑based splits with validation carved from train only; oversample the train subset only if enabled.
- Robustness: Run temporal CV (expanding window) to quantify variance across time; set `train_full_after: true` to refit on full training after CV if needed.
- Feature engineering: Standardize `credit_history_length`; review binning or target encoding for categorical grades if models outside H2O are considered.
- Monitoring: Track thresholded metrics at the fixed validation‑chosen threshold; log PR operating points relevant to business constraints.
