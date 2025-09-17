# Comparison: Kaggle Accepted vs Thesis Data Full

This note documents a reproducible comparison between the Kaggle accepted loans dataset and the thesis canonical full dataset. The goal is to understand differences in size, missingness, dtypes, and row coverage over time.

## Datasets
- Kaggle accepted: `data/raw/full/kaggle_accepted_2007_to_2018Q4.csv`
- Thesis full: `data/raw/full/thesis_data_full.csv`

Both files share the same header (151 columns).

## Line Counts
- Kaggle accepted total lines: 2,260,702 (rows excl. header: 2,260,701)
- Thesis full total lines: 2,104,542 (rows excl. header: 2,104,541)

Delta: thesis has 156,160 fewer rows than Kaggle accepted.

## Column Overlap
- Columns present in both: 151
- Accepted-only: 0
- Thesis-only: 0

## Missing Values per Column
Computed via chunked pandas scan; results saved to `reports/analysis/missing_values_comparison.csv`.

High-level findings:
- Columns with ≥90% missing: accepted=38, thesis=38 (identical set).
- Columns with ≥50% missing: accepted=44, thesis=44 (identical set).
- Largest changes are small:
  - `next_pymnt_d`: +0.92% missing in thesis vs accepted
  - `desc`: −0.41% missing in thesis vs accepted
- Many inquiry/installment-related fields show ~+2.84% higher missingness in thesis, but remain well below the 50% high-missing threshold.

Artifacts:
- Full comparison table: `reports/analysis/missing_values_comparison.csv`
- High-missing (≥50%) detail: `reports/analysis/missing_values_high_50.csv`

## Dtype Comparison
Checked dtypes on a sample of the first 200,000 rows from each file:
- Differences: 0 across 151 columns

Artifact:
- `reports/analysis/dtype_comparison_sample.csv`

## Row-by-Row Comparison (Prefix / Divergence)
- The first 1,000 data rows are identical across files. Header is identical.
- Streaming comparison shows the entire thesis file is a strict prefix of Kaggle accepted:
  - Matched prefix rows: 2,104,541
  - Divergence: thesis ends; accepted continues
  - Last matched `issue_d`: `Nov-2017`

Immediately after divergence:
- The first 100 additional rows in Kaggle accepted all have `issue_d = Nov-2017` (equal to the last matched month). There is not a clean “later dates only” tail.

Tail analysis for the extra 156,160 rows in Kaggle accepted:
- `issue_d` min in tail: `2016-10-01`
- `issue_d` max in tail: `2017-12-01`
- Relative to the last matched month (`2017-11-01`):
  - Later: 2 rows
  - Equal: 14,457 rows
  - Earlier: 141,697 rows
  - Unparsable: 4 rows

Samples:
- First 5 tail `issue_d`: `Nov-2017` x5
- Last 5 tail `issue_d`: `Oct-2016`, `Oct-2016`, `Oct-2016`, `''`, `''`

Artifacts:
- Summary of the immediate divergence window: `reports/analysis/divergence_issue_d_check.txt`
- Tail month distribution summary: `reports/analysis/accepted_tail_issue_d_summary.txt`

## Interpretation
- The thesis dataset appears to be a curated subset (strict prefix) of the Kaggle accepted file, ending at `Nov-2017` when ordered as in the source files. The extra rows in Kaggle accepted are not simply “later months”; they include mostly earlier or the same month (`Nov-2017`), and a very small number in `Dec-2017`.
- High-missing fields are consistent across both datasets, and dtype inference (on a large sample) matches across all columns.

## Reproduction Notes
- All comparisons were run with pandas in chunks to avoid memory pressure.
- Date parsing for `issue_d` used formats: `%b-%Y`, `%Y-%m-%d`, `%Y-%m`.
- Generated artifacts live under `reports/analysis/` as listed above.

