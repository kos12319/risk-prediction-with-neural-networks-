# Column Dictionary — Sample Dataset

This dictionary is generated from a sample of the unzipped dataset (`data/raw/samples/first_10k_rows.csv`).

- Type is inferred from pandas dtypes with date parsing per config, and marked `unknown` when non-null coverage < 5%.
- Missing % is the fraction of rows with null/empty values (0.0–100.0).
- Leaks Target = Yes if the column is listed under `data.leakage_cols` in `configs/default.yaml` (post‑origination information).
- Values: ranges for numeric/date; top categories for strings (up to 10).

| Column | Type | Missing % | Leaks Target | Description | Values |
|---|---:|---:|---:|---|---|
| id | number | 0.0% | No | Unique loan identifier. | range: 3.62e+05 – 6.86e+07 |
| member_id | unknown | 100.0% | No | Unique borrower identifier (internal). |  |
| loan_amnt | number | 0.0% | No | Requested loan amount (USD) at origination. | range: 1e+03 – 3.5e+04 |
| funded_amnt | number | 0.0% | No | Total amount committed by investors (USD). | range: 1e+03 – 3.5e+04 |
| funded_amnt_inv | number | 0.0% | No | Portion of funded amount committed by investors (USD). | range: 1e+03 – 3.5e+04 |
| term | string | 0.0% | No | Loan term in months (e.g., 36, 60). |  36 months,  60 months |
| int_rate | number | 0.0% | No | Interest rate on the loan (%). | range: 5.32 – 29 |
| installment | number | 0.0% | No | Monthly payment owed by the borrower (USD). | range: 30.5 – 1.35e+03 |
| grade | string | 0.0% | No | Lender-assigned credit grade. | B, C, A, D, E, F, G |
| sub_grade | string | 0.0% | No | Lender-assigned sub-grade. | C1, B4, B2, B3, C2, B5, C4, C3, B1, A5 (+25 more) |
| emp_title | string | 5.6% | No | Borrower employment title (free text). | Teacher, Manager, Owner, Registered Nurse, Supervisor, Project Manager, RN, Sales, Driver, manager (+5612 more) |
| emp_length | string | 5.5% | No | Employment length in years (bucketed). | 10+ years, < 1 year, 2 years, 3 years, 1 year, 5 years, 4 years, 8 years, 6 years, 9 years (+1 more) |
| home_ownership | string | 0.0% | No | Home ownership status (e.g., RENT, MORTGAGE, OWN). | MORTGAGE, RENT, OWN |
| annual_inc | number | 0.0% | No | Annual self-reported income (USD). | range: 1.77e+03 – 3.96e+06 |
| verification_status | string | 0.0% | No | Income verification status by the lender. | Source Verified, Not Verified, Verified |
| issue_d | date | 0.0% | No | Loan issue date. | range: 2015-12-01 – 2015-12-01 |
| loan_status | string | 0.0% | No | Loan outcome/target label. | Fully Paid, Charged Off, Current, Late (31-120 days), In Grace Period, Late (16-30 days) |
| pymnt_plan | string | 0.0% | No | Payments plan flag (e.g., y/n). | n, y |
| url | string | 0.0% | No | Listing URL (identifier). | https://lendingclub.com/browse/loanDetail.action?loan_id=68407277, https://lendingclub.com/browse/loanDetail.action?loan_id=68242350, htt... |
| desc | unknown | 100.0% | No | Borrower description (free text). |  |
| purpose | string | 0.0% | No | Borrower-stated loan purpose category. | debt_consolidation, credit_card, home_improvement, other, major_purchase, medical, car, small_business, vacation, moving (+2 more) |
| title | string | 1.3% | No | Loan title (user-entered). | Debt consolidation, Credit card refinancing, Home improvement, Other, Major purchase, Medical expenses, Car financing, Business, Vacation... |
| zip_code | string | 0.0% | No | Borrower ZIP3 region. | 112xx, 945xx, 300xx, 750xx, 606xx, 770xx, 917xx, 331xx, 900xx, 070xx (+779 more) |
| addr_state | string | 0.0% | No | Borrower state or territory. | CA, TX, NY, FL, IL, NJ, GA, VA, OH, PA (+39 more) |
| dti | number | 0.0% | No | Debt-to-income ratio at application time. | range: 0 – 999 |
| delinq_2yrs | number | 0.0% | No | Delinquencies in the past 2 years. | range: 0 – 15 |
| earliest_cr_line | date | 0.0% | No | Date of earliest reported credit line. | range: 1957-01-01 – 2012-11-01 |
| fico_range_low | number | 0.0% | No | Lower bound of FICO score range provided. | range: 660 – 845 |
| fico_range_high | number | 0.0% | No | Upper bound of FICO score range provided. | range: 664 – 850 |
| inq_last_6mths | number | 0.0% | No | Credit inquiries in the last 6 months. | range: 0 – 5 |
| mths_since_last_delinq | number | 48.0% | No | Months since most recent delinquency. | range: 0 – 116 |
| mths_since_last_record | number | 82.2% | No | Months since most recent public record. | range: 0 – 119 |
| open_acc | number | 0.0% | No | Number of open credit lines. | range: 1 – 55 |
| pub_rec | number | 0.0% | No | Number of derogatory public records. | range: 0 – 23 |
| revol_bal | number | 0.0% | No | Total revolving credit balance (USD). | range: 0 – 5.66e+05 |
| revol_util | number | 0.1% | No | Revolving line utilization rate (%). | range: 0 – 134 |
| total_acc | number | 0.0% | No | Total number of credit lines ever opened. | range: 4 – 105 |
| initial_list_status | string | 0.0% | No | Initial listing status of the loan. | w, f |
| out_prncp | number | 0.0% | Yes | Outstanding principal (post-origination; leaky). | range: 0 – 2.27e+04 |
| out_prncp_inv | number | 0.0% | Yes | Outstanding principal for investors (leaky). | range: 0 – 2.27e+04 |
| total_pymnt | number | 0.0% | Yes | Total payments received to date (leaky). | range: 0 – 5.26e+04 |
| total_pymnt_inv | number | 0.0% | Yes | Total payments received to date for investors (leaky). | range: 0 – 5.26e+04 |
| total_rec_prncp | number | 0.0% | No | Total principal received to date. | range: 0 – 3.5e+04 |
| total_rec_int | number | 0.0% | No | Total interest received to date. | range: 0 – 2.33e+04 |
| total_rec_late_fee | number | 0.0% | No | Total late fees received to date. | range: 0 – 591 |
| recoveries | number | 0.0% | Yes | Recoveries received after charge-off (leaky). | range: 0 – 2.22e+04 |
| collection_recovery_fee | number | 0.0% | Yes | Collection recovery fee (post-charge-off; leaky). | range: 0 – 4e+03 |
| last_pymnt_d | string | 0.1% | Yes | Date of last payment (post-origination; leaky). | Jan-2019, Mar-2019, Dec-2018, Feb-2019, Aug-2017, Mar-2017, Nov-2017, Mar-2018, Jun-2017, Sep-2017 (+30 more) |
| last_pymnt_amnt | number | 0.0% | Yes | Amount of last payment (leaky). | range: 0 – 3.61e+04 |
| next_pymnt_d | string | 89.0% | Yes | Scheduled next payment date (leaky). | Apr-2019, Mar-2019 |
| last_credit_pull_d | string | 0.0% | Yes | Last credit pull date (post-origination). | Mar-2019, Dec-2018, Jan-2019, Feb-2019, Jul-2018, Nov-2018, Oct-2018, Aug-2018, Sep-2018, Feb-2017 (+30 more) |
| last_fico_range_high | number | 0.0% | No | Latest reported FICO range upper bound. | range: 0 – 844 |
| last_fico_range_low | number | 0.0% | No | Latest reported FICO range lower bound. | range: 0 – 840 |
| collections_12_mths_ex_med | number | 0.0% | No | Collections in last 12 months excluding medical. | range: 0 – 3 |
| mths_since_last_major_derog | number | 70.7% | No | Months since most recent major derogatory. | range: 0 – 118 |
| policy_code | number | 0.0% | No | Policy code (LC internal). | range: 1 – 1 |
| application_type | string | 0.0% | No | Application type (INDIVIDUAL/JOINT). | Individual, Joint App |
| annual_inc_joint | unknown | 99.4% | No | Joint annual income (if joint application). |  |
| dti_joint | unknown | 99.4% | No | Joint debt-to-income ratio. |  |
| verification_status_joint | unknown | 99.4% | No | Joint income verification status. |  |
| acc_now_delinq | number | 0.0% | No | Number of accounts currently delinquent. | range: 0 – 2 |
| tot_coll_amt | number | 0.0% | No | Total collection amounts ever. | range: 0 – 1.03e+05 |
| tot_cur_bal | number | 0.0% | No | Total current balance excluding mortgage. | range: 0 – 2.13e+06 |
| open_acc_6m | number | 0.0% | No | Open accounts in past 6 months. | range: 0 – 14 |
| open_act_il | number | 0.0% | No | Open active installment accounts. | range: 0 – 34 |
| open_il_12m | number | 0.0% | No | Installment accounts opened in last 12 months. | range: 0 – 10 |
| open_il_24m | number | 0.0% | No | Installment accounts opened in last 24 months. | range: 0 – 19 |
| mths_since_rcnt_il | number | 2.4% | No | Months since most recent installment account. | range: 0 – 338 |
| total_bal_il | number | 0.0% | No | Total balance on installment accounts. | range: 0 – 8.78e+05 |
| il_util | number | 12.5% | No | Installment utilization ratio (%). | range: 0 – 210 |
| open_rv_12m | number | 0.0% | No | Revolving accounts opened in last 12 months. | range: 0 – 22 |
| open_rv_24m | number | 0.0% | No | Revolving accounts opened in last 24 months. | range: 0 – 43 |
| max_bal_bc | number | 0.0% | No | Maximum bankcard balance. | range: 0 – 1.27e+05 |
| all_util | number | 0.0% | No | Balance to credit limit ratio across all trades. | range: 0 – 141 |
| total_rev_hi_lim | number | 0.0% | No | Total revolving high credit/limit. | range: 0 – 5.91e+05 |
| inq_fi | number | 0.0% | No | Personal finance inquiries. | range: 0 – 17 |
| total_cu_tl | number | 0.0% | No | Total credit union trade lines. | range: 0 – 33 |
| inq_last_12m | number | 0.0% | No | Credit inquiries in the last 12 months. | range: 0 – 30 |
| acc_open_past_24mths | number | 0.0% | No | Accounts opened in past 24 months. | range: 0 – 50 |
| avg_cur_bal | number | 0.0% | No | Average current balance per open account. | range: 0 – 2.36e+05 |
| bc_open_to_buy | number | 1.0% | No | Bankcard open to buy (limit minus balance). | range: 0 – 2.64e+05 |
| bc_util | number | 1.1% | No | Bankcard utilization rate (%). | range: 0 – 138 |
| chargeoff_within_12_mths | number | 0.0% | No | Charge-offs within last 12 months. | range: 0 – 2 |
| delinq_amnt | number | 0.0% | No | Past-due amount on delinquent accounts. | range: 0 – 6.5e+04 |
| mo_sin_old_il_acct | number | 2.4% | No | Months since oldest installment account opened. | range: 1 – 429 |
| mo_sin_old_rev_tl_op | number | 0.0% | No | Months since oldest revolving account opened. | range: 6 – 707 |
| mo_sin_rcnt_rev_tl_op | number | 0.0% | No | Months since most recent revolving account opened. | range: 0 – 180 |
| mo_sin_rcnt_tl | number | 0.0% | No | Months since most recent trade line opened. | range: 0 – 180 |
| mort_acc | number | 0.0% | No | Number of mortgage accounts. | range: 0 – 18 |
| mths_since_recent_bc | number | 1.0% | No | Months since most recent bankcard account. | range: 0 – 387 |
| mths_since_recent_bc_dlq | number | 74.4% | No | Months since most recent bankcard delinquency. | range: 0 – 116 |
| mths_since_recent_inq | number | 10.7% | No | Months since most recent inquiry. | range: 0 – 24 |
| mths_since_recent_revol_delinq | number | 64.0% | No | Months since most recent revolving delinquency. | range: 0 – 116 |
| num_accts_ever_120_pd | number | 0.0% | No | Accounts ever 120+ days past due. | range: 0 – 26 |
| num_actv_bc_tl | number | 0.0% | No | Active bankcard trade lines. | range: 0 – 23 |
| num_actv_rev_tl | number | 0.0% | No | Number of active revolving trade lines. | range: 0 – 31 |
| num_bc_sats | number | 0.0% | No | Number of satisfactory bankcard accounts. | range: 0 – 32 |
| num_bc_tl | number | 0.0% | No | Number of bankcard accounts. | range: 0 – 44 |
| num_il_tl | number | 0.0% | No | Number of installment accounts. | range: 0 – 80 |
| num_op_rev_tl | number | 0.0% | No | Open revolving trade lines. | range: 0 – 46 |
| num_rev_accts | number | 0.0% | No | Number of revolving accounts. | range: 2 – 80 |
| num_rev_tl_bal_gt_0 | number | 0.0% | No | Number of revolving trade lines with balance > 0. | range: 0 – 32 |
| num_sats | number | 0.0% | No | Number of satisfactory accounts. | range: 1 – 55 |
| num_tl_120dpd_2m | number | 5.6% | No | Trade lines 120+ days past due in last 2 months. | range: 0 – 1 |
| num_tl_30dpd | number | 0.0% | No | Trade lines 30+ days past due. | range: 0 – 2 |
| num_tl_90g_dpd_24m | number | 0.0% | No | Number of trade lines 90+ days past due in last 24 months. | range: 0 – 13 |
| num_tl_op_past_12m | number | 0.0% | No | Trade lines opened in past 12 months. | range: 0 – 25 |
| pct_tl_nvr_dlq | number | 0.0% | No | Percent of trade lines never delinquent (%). | range: 35 – 100 |
| percent_bc_gt_75 | number | 1.1% | No | Percent of bankcards with utilization > 75%. | range: 0 – 100 |
| pub_rec_bankruptcies | number | 0.0% | No | Number of public record bankruptcies. | range: 0 – 8 |
| tax_liens | number | 0.0% | No | Number of tax liens. | range: 0 – 22 |
| tot_hi_cred_lim | number | 0.0% | No | Total high credit/limit across all accounts. | range: 2.7e+03 – 2.39e+06 |
| total_bal_ex_mort | number | 0.0% | No | Total balance excluding mortgage accounts. | range: 0 – 8.79e+05 |
| total_bc_limit | number | 0.0% | No | Total bankcard credit limit. | range: 0 – 3.03e+05 |
| total_il_high_credit_limit | number | 0.0% | No | Total installment high credit limit. | range: 0 – 5.91e+05 |
| revol_bal_joint | unknown | 100.0% | No | Joint revolving balance. |  |
| sec_app_fico_range_low | unknown | 100.0% | No | Secondary applicant FICO range low. |  |
| sec_app_fico_range_high | unknown | 100.0% | No | Secondary applicant FICO range high. |  |
| sec_app_earliest_cr_line | unknown | 100.0% | No | Secondary applicant earliest credit line date. |  |
| sec_app_inq_last_6mths | unknown | 100.0% | No | Secondary applicant inquiries in last 6 months. |  |
| sec_app_mort_acc | unknown | 100.0% | No | Secondary applicant mortgage accounts. |  |
| sec_app_open_acc | unknown | 100.0% | No | Secondary applicant open accounts. |  |
| sec_app_revol_util | unknown | 100.0% | No | Secondary applicant revolving utilization (%). |  |
| sec_app_open_act_il | unknown | 100.0% | No | Secondary applicant open active installment accounts. |  |
| sec_app_num_rev_accts | unknown | 100.0% | No | Secondary applicant revolving accounts. |  |
| sec_app_chargeoff_within_12_mths | unknown | 100.0% | No | Secondary applicant charge-offs within 12 months. |  |
| sec_app_collections_12_mths_ex_med | unknown | 100.0% | No | Secondary applicant collections (ex medical) within 12 months. |  |
| sec_app_mths_since_last_major_derog | unknown | 100.0% | No | Secondary applicant months since major derogatory. |  |
| hardship_flag | string | 0.0% | Yes | Hardship program flag (post-origination; leaky). | N, Y |
| hardship_type | unknown | 99.2% | Yes | Hardship program type (leaky). |  |
| hardship_reason | unknown | 99.2% | Yes | Reason for hardship (leaky). |  |
| hardship_status | unknown | 99.2% | Yes | Hardship status (leaky). |  |
| deferral_term | unknown | 99.2% | No | Hardship deferral term length. |  |
| hardship_amount | unknown | 99.2% | Yes | Hardship amount (leaky). |  |
| hardship_start_date | unknown | 99.2% | Yes | Hardship start date (leaky). |  |
| hardship_end_date | unknown | 99.2% | Yes | Hardship end date (leaky). |  |
| payment_plan_start_date | unknown | 99.2% | Yes | Payment plan start date (leaky). |  |
| hardship_length | unknown | 99.2% | Yes | Length of hardship plan (leaky). |  |
| hardship_dpd | unknown | 99.2% | Yes | Hardship days past due (leaky). |  |
| hardship_loan_status | unknown | 99.2% | Yes | Loan status under hardship (leaky). |  |
| orig_projected_additional_accrued_interest | unknown | 99.3% | Yes | Projected additional accrued interest (post-origination; leaky). |  |
| hardship_payoff_balance_amount | unknown | 99.2% | Yes | Hardship payoff balance amount (leaky). |  |
| hardship_last_payment_amount | unknown | 99.2% | Yes | Hardship last payment amount (leaky). |  |
| disbursement_method | string | 0.0% | No | Method of disbursement (e.g., CASH, DIRECT_PAY). | Cash |
| debt_settlement_flag | string | 0.0% | Yes | Debt settlement flag (post-origination; leaky). | N, Y |
| debt_settlement_flag_date | unknown | 97.2% | Yes | Date of debt settlement flag (leaky). |  |
| settlement_status | unknown | 97.2% | Yes | Debt settlement status (leaky). |  |
| settlement_date | unknown | 97.2% | Yes | Debt settlement date (leaky). |  |
| settlement_amount | unknown | 97.2% | Yes | Settlement amount (leaky). |  |
| settlement_percentage | unknown | 97.2% | Yes | Settlement percentage (leaky). |  |
| settlement_term | unknown | 97.2% | Yes | Settlement term (leaky). |  |