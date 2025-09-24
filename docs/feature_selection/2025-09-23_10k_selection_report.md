# Feature Selection Report – 10k Sample (2025-09-23)

## Context
- Dataset: `data/raw/samples/thesis_data_sample_10k.csv`
- Split: time-based on `issue_d`, 20% holdout test from latest vintages
- Positive class: `Charged Off` (`eval.pos_label=0`)
- Winsorization: disabled for all runs to ensure feature selection pipelines expose `get_feature_names_out`

Three selection variants were evaluated:
1. **Mutual Information (MI)** filter on the full provider-agnostic feature list
2. **L1 logistic (independent)** embedded selection on the full list
3. **L1-on-MI** applied only to the 15-feature MI shortlist

Each subset was trained with the standard PyTorch MLP backend (`layers=[256,128,64,32]`, `epochs=30`, early stopping).

## Selected Feature Sets

### MI (15 features)
`addr_state`, `term`, `verification_status`, `purpose`, `emp_length`, `home_ownership`, `fico_range_low`, `dti`, `fico_range_high`, `num_rev_tl_bal_gt_0`, `tot_hi_cred_lim`, `num_bc_sats`, `loan_amnt`, `total_bc_limit`, `bc_open_to_buy`

Artifacts: `selection_runs/run_<timestamp>_select/mi/` (ranking, curve, results JSON)

### L1 Logistic (independent, 12 features)
`addr_state`, `term`, `home_ownership`, `emp_length`, `verification_status`, `dti`, `purpose`, `inq_last_6mths`, `loan_amnt`, `fico_range_high`, `fico_range_low`, `revol_bal`

Artifacts: `selection_runs/run_<timestamp>_select/l1/`

### L1-on-MI (chained, 12 features)
Applied L1 logistic to the MI shortlist. Resulting subset:
`addr_state`, `term`, `home_ownership`, `verification_status`, `dti`, `purpose`, `emp_length`, `total_bc_limit`, `fico_range_high`, `fico_range_low`, `loan_amnt`, `bc_open_to_buy`

Artifacts: `selection_runs/run_<timestamp>_select/l1/` (re-run with `CONFIG=configs/pytorch_default.yaml`)

## Training Outcomes

| Config | Encoded Features | ROC AUC (test) | Avg Precision | Threshold | W&B Run |
| --- | --- | --- | --- | --- | --- |
| `configs/pytorch_default.yaml` | 92 | 0.657 | 0.340 | 0.431 | [4vbv75sy](https://wandb.ai/petr-encode-peterlamb/loan-risk-mlp/runs/4vbv75sy) |
| Variant L1 subset | 89 | **0.723** | **0.407** | 0.483 | [4wd8uyoe](https://wandb.ai/petr-encode-peterlamb/loan-risk-mlp/runs/4wd8uyoe) |
| Variant L1 on MI subset | 89 | 0.676 | 0.348 | 0.434 | [ewhrixls](https://wandb.ai/petr-encode-peterlamb/loan-risk-mlp/runs/ewhrixls) |

> Encoded feature counts include one-hot expansions; numerical columns remain single-output.

## Observations

- The independent L1 subset outperforms MI by ~0.07 ROC AUC and ~0.07 AP while retaining a compact 12-feature set.
- Chaining MI → L1 trims two MI features (`num_rev_tl_bal_gt_0`, `tot_hi_cred_lim`) but introduces `total_bc_limit` and `bc_open_to_buy`; performance drops relative to the independent L1 run, suggesting MI’s filter removed signals useful to the sparse logistic model.
- Positive-class recall at the selected thresholds: 0.493 (MI), 0.659 (L1), 0.587 (L1-on-MI). The L1 subset provides the best balance of recall and precision.
- Recommendation: use the independent L1 subset as the starting point for downstream modeling; revisit MI only for exploratory ranking or if you need a fallback when embedded methods fail to converge.

## Repro Steps

```
make select CONFIG=configs/pytorch_default.yaml METHOD=mi
make select CONFIG=configs/pytorch_default.yaml METHOD=l1
make select CONFIG=configs/pytorch_default.yaml METHOD=l1  # chained run
make train CONFIG=configs/pytorch/default_mi_subset.yaml
make train CONFIG=configs/pytorch/default_l1_subset.yaml
make train CONFIG=configs/pytorch/default_l1_on_mi_subset.yaml
```

All configs and artifacts live under `configs/` and `local_runs/` as referenced above.
