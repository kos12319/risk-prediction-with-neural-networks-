# Data Directory

Recommended layout for datasets used in this project:

- raw/
  - full/ — canonical unzipped datasets (ignored by git)
  - archives/ — original downloads, zips (tracked via Git LFS by pattern `*.zip`)
  - samples/ — small demo CSVs for quick runs (tracked, no LFS)
- processed/ — cached splits or derived summaries (ignored by git)

Examples:
- Sample (default config): `data/raw/samples/first_10k_rows.csv`
- Full (local only, ignored): `data/raw/full/loans_full.csv`
- Archives (LFS): `data/raw/archives/loans_full_2024-09.zip`

Update `configs/*.yaml` → `data.csv_path` to point to the file you want to train on.

