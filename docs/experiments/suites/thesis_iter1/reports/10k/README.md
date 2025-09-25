# Thesis Iter 1 — Dataset 10k Report

## Summary
- Winner run: `run_20250925_023120`
- Winner metrics: Avg Precision 0.4601, ROC AUC 0.7591, threshold 0.1487
- Best feature set: 43 features (broad + `int_rate`, `grade`, `sub_grade`, `installment`)

## Runs and Metrics
| Run | Features | ROC AUC | Avg Precision | Threshold |
|---|---:|---:|---:|---:|
| run_20250925_023120 | 43 | 0.7590882169326566 | 0.46008015396470603 | 0.1487004878593114 |
| run_20250925_023823 | 16 | 0.7523050695522723 | 0.4263725375866705 | 0.2033690620838399 |
| run_20250925_021716 | 39 | 0.7467304829440861 | 0.45115294554573937 | 0.1314934263914062 |
| run_20250925_022418 | 12 | 0.7360005042339646 | 0.42057668867476156 | 0.3879084587097168 |

## Feature Sets (by run)
- run_20250925_021716:
  loan_amnt  term  purpose  emp_length  annual_inc  verification_status  home_ownership  addr_state  dti  delinq_2yrs  pub_rec  inq_last_6mths  open_acc  total_acc  revol_bal  revol_util  fico_range_low  fico_range_high  mort_acc  total_rev_hi_lim  bc_open_to_buy  bc_util  percent_bc_gt_75  tot_cur_bal  tot_hi_cred_lim  total_bal_ex_mort  total_bc_limit  total_il_high_credit_limit  num_rev_accts  num_rev_tl_bal_gt_0  num_il_tl  num_bc_tl  num_bc_sats  num_actv_rev_tl  num_tl_90g_dpd_24m  pub_rec_bankruptcies  tax_liens  issue_d  earliest_cr_line
- run_20250925_022418:
  addr_state  term  home_ownership  emp_length  verification_status  dti  purpose  inq_last_6mths  loan_amnt  fico_range_high  fico_range_low  revol_bal
- run_20250925_023120:
  loan_amnt  term  purpose  emp_length  annual_inc  verification_status  home_ownership  addr_state  int_rate  grade  sub_grade  installment  dti  delinq_2yrs  pub_rec  inq_last_6mths  open_acc  total_acc  revol_bal  revol_util  fico_range_low  fico_range_high  mort_acc  total_rev_hi_lim  bc_open_to_buy  bc_util  percent_bc_gt_75  tot_cur_bal  tot_hi_cred_lim  total_bal_ex_mort  total_bc_limit  total_il_high_credit_limit  num_rev_accts  num_rev_tl_bal_gt_0  num_il_tl  num_bc_tl  num_bc_sats  num_actv_rev_tl  num_tl_90g_dpd_24m  pub_rec_bankruptcies  tax_liens  issue_d  earliest_cr_line
- run_20250925_023823:
  addr_state  term  home_ownership  emp_length  verification_status  dti  purpose  inq_last_6mths  loan_amnt  fico_range_high  fico_range_low  revol_bal  int_rate  grade  sub_grade  installment

## Important Variables (winner: GBM family)
- Top 10 by GBM importance: num__int_rate, cat__term_ 60 months, num__dti, num__loan_amnt, num__income_to_loan_ratio, cat__term_ 36 months, cat__grade_E, num__num_actv_rev_tl, num__credit_history_length, num__total_rev_hi_lim

## Figures
- PR curve: `figures/pr_curve.png`
- ROC curve: `figures/roc_curve.png`
- VarImp (winners heatmap): `figures/h2o_varimp_heatmap_winners.png`
- Leaderboard PR: `figures/h2o_leaderboard_pr.png`
- Leaderboard ROC: `figures/h2o_leaderboard_roc.png`
- Feature-set AUCPR comparison: `figures/aupr_by_feature_set.svg`

## Notes & Suggestions
- Adding `int_rate/grade/sub_grade` consistently improves AUCPR at 10k scale.
- Consider exploring additional engineered features (e.g., `income_to_loan_ratio`, credit history length) which already show up as important.
- Use the same time-based split and thresholding (Youden J on validation) for comparability.
