# Thesis Iter 1 — Dataset 100k Report

## Summary
- Winner run: `run_20250925_032002`
- Winner metrics: Avg Precision 0.4524, ROC AUC 0.7435, threshold 0.1783
- Best feature set: 43 features (broad + `int_rate`, `grade`, `sub_grade`, `installment`)

## Runs and Metrics
| Run | Features | ROC AUC | Avg Precision | Threshold |
|---|---:|---:|---:|---:|
| run_20250925_032002 | 43 | 0.7434524534949423 | 0.45236984423780524 | 0.1782799363136291 |
| run_20250925_033737 | 16 | 0.7392301407711371 | 0.44521227426410176 | 0.1922218799591064 |
| run_20250925_030244 | 12 | 0.7252230936183267 | 0.4269055804186448 | 0.1652029752731323 |
| run_20250925_024526 | 39 | 0.730419404159142 | 0.44189009951229435 | 0.1709468960762024 |

## Feature Sets (by run)
- run_20250925_024526:
  loan_amnt  term  purpose  emp_length  annual_inc  verification_status  home_ownership  addr_state  dti  delinq_2yrs  pub_rec  inq_last_6mths  open_acc  total_acc  revol_bal  revol_util  fico_range_low  fico_range_high  mort_acc  total_rev_hi_lim  bc_open_to_buy  bc_util  percent_bc_gt_75  tot_cur_bal  tot_hi_cred_lim  total_bal_ex_mort  total_bc_limit  total_il_high_credit_limit  num_rev_accts  num_rev_tl_bal_gt_0  num_il_tl  num_bc_tl  num_bc_sats  num_actv_rev_tl  num_tl_90g_dpd_24m  pub_rec_bankruptcies  tax_liens  issue_d  earliest_cr_line
- run_20250925_030244:
  addr_state  term  home_ownership  emp_length  verification_status  dti  purpose  inq_last_6mths  loan_amnt  fico_range_high  fico_range_low  revol_bal
- run_20250925_032002:
  loan_amnt  term  purpose  emp_length  annual_inc  verification_status  home_ownership  addr_state  int_rate  grade  sub_grade  installment  dti  delinq_2yrs  pub_rec  inq_last_6mths  open_acc  total_acc  revol_bal  revol_util  fico_range_low  fico_range_high  mort_acc  total_rev_hi_lim  bc_open_to_buy  bc_util  percent_bc_gt_75  tot_cur_bal  tot_hi_cred_lim  total_bal_ex_mort  total_bc_limit  total_il_high_credit_limit  num_rev_accts  num_rev_tl_bal_gt_0  num_il_tl  num_bc_tl  num_bc_sats  num_actv_rev_tl  num_tl_90g_dpd_24m  pub_rec_bankruptcies  tax_liens  issue_d  earliest_cr_line
- run_20250925_033737:
  addr_state  term  home_ownership  emp_length  verification_status  dti  purpose  inq_last_6mths  loan_amnt  fico_range_high  fico_range_low  revol_bal  int_rate  grade  sub_grade  installment

## Important Variables (winner: GBM family)
- Top 10 by GBM importance: num__int_rate, cat__term_ 36 months, cat__term_ 60 months, num__dti, cat__grade_A, num__num_rev_tl_bal_gt_0, num__tot_hi_cred_lim, cat__grade_B, num__loan_amnt, cat__grade_C

## Figures
- PR curve: `figures/pr_curve.png`
- ROC curve: `figures/roc_curve.png`
- VarImp (winners heatmap): `figures/h2o_varimp_heatmap_winners.png`
- Leaderboard PR: `figures/h2o_leaderboard_pr.png`
- Leaderboard ROC: `figures/h2o_leaderboard_roc.png`
- Feature-set AUCPR comparison: `figures/aupr_by_feature_set.svg`

## Notes & Suggestions
- At 100k, enriched features with pricing/grade information outperform leaner baselines.
- Consider partial dependence for top drivers (enabled in config) to illustrate monotonic effects for `int_rate`, `dti`, and term.
- Verify no leakage: all features listed are available at origination; maintain `data.drop_leakage: true`.
