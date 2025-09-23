# Rejected Applications — Research Notes

This memo captures how the LendingClub rejected-applications file (`kaggle_rejected_2007_to_2018Q4.csv`) can support the thesis without committing to pipeline changes.

## Schema Snapshot
- Columns: `Amount Requested`, `Application Date`, `Loan Title`, `Risk_Score`, `Debt-To-Income Ratio`, `Zip Code`, `State`, `Employment Length`, `Policy Code`.
- No repayment outcomes; feature coverage is coarse compared with the 150+ fields available in the accepted-loans dataset.
- Overlaps by concept (loan amount, DTI, state, employment length, policy code) but not by exact column names; `Risk_Score` is unique to this table.

## Potential Uses
- **Approval-bias EDA**: Compare distribution of key origination features (loan amount, DTI, employment length, geography, risk score) between rejected and accepted applications to document lender screening effects and contextualize the feature-selection story.
- **Feature-subset stress test**: Check whether the stability-selected subset still captures the dimensions that separate accepted vs rejected cohorts, indicating portability across the approval boundary.
- **Reject-inference experiments**: Prototype semi-supervised/propensity-weighted augmentation to infer outcomes for rejected rows and assess whether neural models with compact feature sets maintain calibration and AUC.
- **Fairness diagnostics**: Evaluate whether the approval process disproportionately filters certain demographics or regions; investigate how the proposed feature subset behaves under those shifts.
- **Monitoring signal**: Use drift in the rejected population as an early warning for policy or macro changes that might invalidate trained models or their calibration assumptions.

## Integration Considerations
- Keep rejected-data analyses separate from the core training pipeline to avoid accidental leakage.
- When referencing these results in the thesis, frame them as auxiliary evidence supporting portability, robustness, and responsible feature selection.
- If reject inference is pursued, document assumptions carefully (e.g., MAR vs MNAR) and quantify sensitivity to inferred labels.

