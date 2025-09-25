# Thesis Iter 1 — Full Dataset Report

## Summary
- Winner run: `run_20250925_070714`
- Winner metrics: Avg Precision 0.3934, ROC AUC 0.7093, threshold 0.1765 (Youden J on validation; `eval.pos_label=0`)
- Best feature set: 43 features (broad + `int_rate`, `grade`, `sub_grade`, `installment`)

## Runs and Metrics
| Run | Features | ROC AUC | Avg Precision | Threshold |
|---|---:|---:|---:|---:|
| run_20250925_070714 | 43 | 0.7093281869426106 | 0.3934406581176853 | 0.1764856954498019 |
| run_20250925_035452 | 39 | 0.7001630030430313 | 0.38393766338968977 | 0.1648596611973921 |
| run_20250925_053155 | 12 | 0.6815307488949233 | 0.3655866418292786 | 0.1724987030029297 |
| run_20250925_084408 | 16 | 0.699899888201011 | 0.382459609692605 | 0.1643866832437756 |

Interpretation:
- Enriching the baseline with pricing/grade (`int_rate`, `grade/sub_grade`, `installment`) yields the best AUCPR and ROC AUC.
- The compact 12–16 feature sets trail by ~0.01–0.03 AUCPR; the 39-feature set without pricing/grade underperforms the 43-feature set.

## Feature Sets (by run)
- run_20250925_035452:
  loan_amnt  term  purpose  emp_length  annual_inc  verification_status  home_ownership  addr_state  dti  delinq_2yrs  pub_rec  inq_last_6mths  open_acc  total_acc  revol_bal  revol_util  fico_range_low  fico_range_high  mort_acc  total_rev_hi_lim  bc_open_to_buy  bc_util  percent_bc_gt_75  tot_cur_bal  tot_hi_cred_lim  total_bal_ex_mort  total_bc_limit  total_il_high_credit_limit  num_rev_accts  num_rev_tl_bal_gt_0  num_il_tl  num_bc_tl  num_bc_sats  num_actv_rev_tl  num_tl_90g_dpd_24m  pub_rec_bankruptcies  tax_liens  issue_d  earliest_cr_line
- run_20250925_053155:
  addr_state  term  home_ownership  emp_length  verification_status  dti  purpose  inq_last_6mths  loan_amnt  fico_range_high  fico_range_low  revol_bal
- run_20250925_070714:
  loan_amnt  term  purpose  emp_length  annual_inc  verification_status  home_ownership  addr_state  int_rate  grade  sub_grade  installment  dti  delinq_2yrs  pub_rec  inq_last_6mths  open_acc  total_acc  revol_bal  revol_util  fico_range_low  fico_range_high  mort_acc  total_rev_hi_lim  bc_open_to_buy  bc_util  percent_bc_gt_75  tot_cur_bal  tot_hi_cred_lim  total_bal_ex_mort  total_bc_limit  total_il_high_credit_limit  num_rev_accts  num_rev_tl_bal_gt_0  num_il_tl  num_bc_tl  num_bc_sats  num_actv_rev_tl  num_tl_90g_dpd_24m  pub_rec_bankruptcies  tax_liens  issue_d  earliest_cr_line
- run_20250925_084408:
  addr_state  term  home_ownership  emp_length  verification_status  dti  purpose  inq_last_6mths  loan_amnt  fico_range_high  fico_range_low  revol_bal  int_rate  grade  sub_grade  installment

## Important Variables (winner: GBM family)
- Top 10 by GBM importance: num__int_rate, cat__term_ 60 months, cat__grade_A, cat__term_ 36 months, cat__grade_B, num__dti, num__income_to_loan_ratio, num__fico_avg, num__mort_acc, num__annual_inc

## Threshold & Confusion (test at fixed threshold)
- Threshold: 0.1765 (selected on validation via Youden J)
- Confusion (test): tp=36227, tn=129969, fp=68284, fn=19876
- Rates: TPR/Recall=0.646, FPR=0.344, Precision=0.347

## Figures
- PR curve: `figures/pr_curve.png`
- ROC curve: `figures/roc_curve.png`
- VarImp (winners heatmap): `figures/h2o_varimp_heatmap_winners.png`
- Leaderboard PR: `figures/h2o_leaderboard_pr.png`
- Leaderboard ROC: `figures/h2o_leaderboard_roc.png`
- Feature-set AUCPR comparison: `figures/aupr_by_feature_set.svg`

## Recommendations
- Keep the 43-feature set as default for full data; pricing/grade variables are consistently top drivers.
- Consider deriving `credit_history_length = issue_d - earliest_cr_line` (appears in some runs as important) explicitly and validating its stability across vintages.
- Explore calibration (e.g., Platt/Isotonic) if decision thresholds will be tuned for downstream policies; keep threshold selection on validation per protocol.
- Monitor recent vintages for right-censoring; confirm the time split ensures older→train, newer→test and that validation is carved from train only.
