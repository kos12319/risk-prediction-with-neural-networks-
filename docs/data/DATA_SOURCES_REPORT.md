# Data Sources Overview

This note captures the LendingClub datasets currently staged in the repository and how they feed the training and evaluation pipelines. Paths are relative to the project root.

## Accepted Loans Cohort
- **Primary file**: `data/raw/full/thesis_data_full.csv`
  - Cleaned, thesis-ready subset of funded applications covering vintages 2007-06 through 2018-12.
  - Serves as the default full dataset for PyTorch runs when `data.csv_path` points to the `full/` directory (see `configs/*.yaml`).
- **Original download**: `data/raw/full/kaggle_accepted_2007_to_2018Q4.csv`
  - Kaggle source file with lender-supplied fields at origination and final repayment outcomes.
  - Retained for traceability; preprocessing scripts derive the thesis-ready CSV from this asset.
- **Archive**: `data/raw/archives/kaggle_accepted_2007_to_2018Q4.csv.gz`
  - Git LFS–tracked compressed copy of the raw Kaggle export; use `git lfs pull` before unpacking.
- **Legacy bundle**: `data/raw/archives/full data set.zip`
  - Original ZIP uploaded during early exploration; contains the same accepted-loans CSV and ancillary documentation.

## Rejected Applications Cohort
- **Primary file**: `data/raw/full/kaggle_rejected_2007_to_2018Q4.csv`
  - LendingClub applications that were declined; feature coverage is limited (no repayment outcomes).
  - Referenced in `docs/thesis/REJECTED_DATASET_NOTES.md` for reject-inference research; not part of the production training pipeline.
- **Archive**: `data/raw/archives/kaggle_rejected_2007_to_2018Q4.csv.gz`
  - Compressed Kaggle export tracked via Git LFS to preserve provenance.

## Sample Subsets (Quick Runs)
All samples live under `data/raw/samples/` and are safe to version-control.
- `thesis_data_sample_100.csv`, `thesis_data_sample_1k.csv`, `thesis_data_sample_10k.csv`
  - Row-limited slices of the accepted-loans cohort for smoke tests, CI-friendly experiments, or documentation figures.
- `thesis_data_sample_100k.csv`
  - Larger slice kept out of Git history; see the LFS-tracked archive `thesis_data_sample_100k.zip` alongside it.
- Update configs by setting `data.csv_path` to the desired sample to reproduce lightweight runs.

## Processed Artifacts
- `data/processed/missing_data_summary.csv`
  - Profiling output summarizing missingness by feature, used to inform feature-selection and leakage checks.
  - Regenerated via the preprocessing pipeline; not meant for direct model training.

## Operational Notes
- Archives under `data/raw/archives/` are the only large binaries checked into Git thanks to LFS; unzip into `data/raw/full/` before use.
- Working CSVs inside `data/raw/full/` are `.gitignored` to avoid large diffs while enabling local experimentation.
- When adding new sources, mirror this structure: archive in `archives/`, canonical CSV in `full/`, and optional thin samples in `samples/`.
