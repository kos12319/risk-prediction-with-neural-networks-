# Thesis Iter 1 — Dataset 1k Report

## Summary
- Winner run: `run_20250925_020521`
- Winner metrics: Avg Precision 0.3148, ROC AUC 0.7313, threshold 0.4712
- Best feature set: 39 features (broad credit + utilization + history; no `int_rate/grade`)

## Runs and Metrics
| Run | Features | ROC AUC | Avg Precision | Threshold |
|---|---:|---:|---:|---:|
| run_20250925_020521 | 39 | 0.7312854930304594 | 0.31476401590740744 | 0.4711589686860214 |
| run_20250925_021417 | 16 | 0.6486835312338668 | 0.25420883588392945 | 0.338928820992398 |
| run_20250925_021119 | 43 | 0.5921528136293237 | 0.21171488857916343 | 0.0286124909824853 |
| run_20250925_020819 | 12 | 0.64945792462571 | 0.23796242039799867 | 0.1409781570677393 |

## Feature Sets (by run)
- run_20250925_020521:
  loan_amnt  term  purpose  emp_length  annual_inc  verification_status  home_ownership  addr_state  dti  delinq_2yrs  pub_rec  inq_last_6mths  open_acc  total_acc  revol_bal  revol_util  fico_range_low  fico_range_high  mort_acc  total_rev_hi_lim  bc_open_to_buy  bc_util  percent_bc_gt_75  tot_cur_bal  tot_hi_cred_lim  total_bal_ex_mort  total_bc_limit  total_il_high_credit_limit  num_rev_accts  num_rev_tl_bal_gt_0  num_il_tl  num_bc_tl  num_bc_sats  num_actv_rev_tl  num_tl_90g_dpd_24m  pub_rec_bankruptcies  tax_liens  issue_d  earliest_cr_line
- run_20250925_020819:
  addr_state  term  home_ownership  emp_length  verification_status  dti  purpose  inq_last_6mths  loan_amnt  fico_range_high  fico_range_low  revol_bal
- run_20250925_021119:
  loan_amnt  term  purpose  emp_length  annual_inc  verification_status  home_ownership  addr_state  int_rate  grade  sub_grade  installment  dti  delinq_2yrs  pub_rec  inq_last_6mths  open_acc  total_acc  revol_bal  revol_util  fico_range_low  fico_range_high  mort_acc  total_rev_hi_lim  bc_open_to_buy  bc_util  percent_bc_gt_75  tot_cur_bal  tot_hi_cred_lim  total_bal_ex_mort  total_bc_limit  total_il_high_credit_limit  num_rev_accts  num_rev_tl_bal_gt_0  num_il_tl  num_bc_tl  num_bc_sats  num_actv_rev_tl  num_tl_90g_dpd_24m  pub_rec_bankruptcies  tax_liens  issue_d  earliest_cr_line
- run_20250925_021417:
  addr_state  term  home_ownership  emp_length  verification_status  dti  purpose  inq_last_6mths  loan_amnt  fico_range_high  fico_range_low  revol_bal  int_rate  grade  sub_grade  installment

## Important Variables (winner: GBM family)
- Top 10 by GBM importance: cat__term_ 60 months, cat__term_ 36 months, num__dti, num__annual_inc, num__tot_hi_cred_lim, num__bc_util, num__income_to_loan_ratio, num__revol_util, num__fico_range_high, num__inq_last_6mths

## Figures
- PR curve: `figures/pr_curve.png`
- ROC curve: `figures/roc_curve.png`
- VarImp (winners heatmap): `figures/h2o_varimp_heatmap_winners.png`
- Leaderboard PR: `figures/h2o_leaderboard_pr.png`
- Leaderboard ROC: `figures/h2o_leaderboard_roc.png`
- Feature-set AUCPR comparison: `figures/aupr_by_feature_set.svg`

## Notes & Suggestions
- On 1k, the broader 39-feature set wins; adding `int_rate/grade` with many dummies appears to overfit at this scale.
- For tiny samples, prefer compact, high-signal features and stronger regularization; consider enabling `automl.balance_classes` (already on) and limiting max models.
- Validate stability via temporal CV if runtime permits (`split.cv.enabled: true`) and report aggregated metrics.
