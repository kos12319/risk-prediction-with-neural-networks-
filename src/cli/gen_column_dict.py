from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.cli._bootstrap import apply_safe_env


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _fmt_float(v: float) -> str:
    try:
        return f"{float(v):.3g}"
    except Exception:
        return str(v)


def _extract_existing_descriptions(md_path: Path) -> Dict[str, str]:
    if not md_path.exists():
        return {}
    desc_map: Dict[str, str] = {}
    for ln in md_path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("| ") and ln.count("|") >= 4:
            parts = [p.strip() for p in ln.strip().split("|")][1:-1]
            # Support both 5-col and 6-col tables; description at index -2 for both
            if len(parts) >= 4:
                col = parts[0]
                # For 6-col table: [Column, Type, Missing %, Leaks, Description, Values]
                # For 5-col table: [Column, Type, Leaks, Description, Values]
                desc = parts[-2] if len(parts) >= 5 else parts[3]
                if desc and col not in desc_map:
                    desc_map[col] = desc
    return desc_map


# Minimal default descriptions for common fields
DEFAULT_DESC: Dict[str, str] = {
    "id": "Unique loan identifier.",
    "member_id": "Unique borrower identifier (internal).",
    "loan_amnt": "Requested loan amount (USD) at origination.",
    "funded_amnt": "Total amount committed by investors (USD).",
    "funded_amnt_inv": "Portion of funded amount committed by investors (USD).",
    "term": "Loan term length in months (e.g., 36, 60).",
    "int_rate": "Interest rate on the loan (%).",
    "installment": "Monthly installment amount (USD).",
    "grade": "Lender-assigned credit grade.",
    "sub_grade": "Lender-assigned sub-grade.",
    "emp_title": "Borrower employment title (free text).",
    "emp_length": "Employment length (bucketed years).",
    "home_ownership": "Home ownership status (RENT/MORTGAGE/OWN/OTHER).",
    "annual_inc": "Annual self-reported income (USD).",
    "verification_status": "Income verification status by the lender.",
    "issue_d": "Loan issue date.",
    "earliest_cr_line": "Date of earliest reported credit line.",
    "loan_status": "Loan outcome/target label.",
    "purpose": "Borrower-stated loan purpose category.",
    "zip_code": "Borrower ZIP3 region.",
    "addr_state": "Borrower state or territory.",
    "dti": "Debt-to-income ratio at application time.",
}


def generate_column_dictionary(
    csv_path: Path,
    parse_dates: List[str],
    leakage_cols: List[str],
    out_path: Path,
    sample_rows: int | None = None,
    sparse_unknown_threshold: float = 0.05,
) -> None:
    # Import heavy dependency lazily after env setup
    import pandas as pd  # type: ignore
    parse_dates = list(parse_dates or [])
    leakage_set = set(leakage_cols or [])

    # Read CSV
    try:
        df = pd.read_csv(csv_path, low_memory=False, parse_dates=parse_dates, nrows=sample_rows)
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False, nrows=sample_rows)
    n = len(df)

    # Reuse existing descriptions if present
    existing_desc = _extract_existing_descriptions(out_path)

    rows_md: List[str] = []
    header = (
        "# Column Dictionary — Sample Dataset\n\n"
        f"This dictionary is generated from a sample of the unzipped dataset (`{csv_path.as_posix()}`).\n\n"
        "- Type is inferred from pandas dtypes with date parsing per config, and marked `unknown` when non-null coverage < 5%.\n"
        "- Missing % is the fraction of rows with null/empty values (0.0–100.0).\n"
        "- Leaks Target = Yes if the column is listed under `data.leakage_cols` in `configs/default.yaml` (post‑origination information).\n"
        "- Values: ranges for numeric/date; top categories for strings (up to 10).\n\n"
        "| Column | Type | Missing % | Leaks Target | Description | Values |\n"
        "|---|---:|---:|---:|---|---|\n"
    )
    rows_md.append(header)

    for col in df.columns:
        s = df[col]
        missing_rate = float(s.isna().mean()) if n else 0.0
        non_empty = s.dropna()

        # Type detection
        if pd.api.types.is_datetime64_any_dtype(s):
            ctype = "date"
        elif pd.api.types.is_numeric_dtype(s):
            ctype = "number"
        else:
            ctype = "string"
        if n > 0 and (non_empty.shape[0] / n) < sparse_unknown_threshold:
            ctype = "unknown"

        # Values summary
        values = ""
        if ctype == "number" and not non_empty.empty:
            mn, mx = non_empty.min(), non_empty.max()
            values = f"range: {_fmt_float(mn)} – {_fmt_float(mx)}"
        elif ctype == "date" and not non_empty.empty:
            try:
                mn, mx = non_empty.min(), non_empty.max()
                values = f"range: {mn.date().isoformat()} – {mx.date().isoformat()}"
            except Exception:
                values = ""
        elif ctype == "string" and not non_empty.empty:
            vc = non_empty.value_counts(dropna=True)
            cats = [str(k) for k in vc.index[:10]]
            extra = f" (+{vc.shape[0] - len(cats)} more)" if vc.shape[0] > len(cats) else ""
            values = (", ".join(cats) + extra)
            if len(values) > 140:
                values = values[:137] + "..."

        leaks = "Yes" if col in leakage_set else "No"
        desc = existing_desc.get(col) or DEFAULT_DESC.get(col) or (col.replace("_", " ") + ".")

        miss_pct = f"{missing_rate*100:.1f}%"
        cell_desc = desc.replace("|", "/")
        cell_vals = (values or "").replace("|", "/")
        rows_md.append(f"| {col} | {ctype} | {miss_pct} | {leaks} | {cell_desc} | {cell_vals} |\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(rows_md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a column dictionary Markdown table from a CSV")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config with data settings")
    parser.add_argument("--csv", type=str, default="", help="Override CSV path (else uses config data.csv_path)")
    parser.add_argument("--out", type=str, default="docs/data/COLUMN_DICTIONARY.md", help="Output Markdown path")
    parser.add_argument("--rows", type=int, default=5000, help="Sample row count (0 = all)")
    parser.add_argument("--sparse_unknown_threshold", type=float, default=0.05, help="Mark type=unknown if non-null coverage below this fraction")
    args = parser.parse_args()

    # Apply safe env before importing heavy libs
    apply_safe_env()

    cfg = _load_config(Path(args.config))
    data_cfg = cfg.get("data", {})
    csv_path = Path(args.csv) if args.csv else Path(data_cfg.get("csv_path", ""))
    if not csv_path:
        raise SystemExit("CSV path not provided and not found in config data.csv_path")

    parse_dates = list(data_cfg.get("parse_dates", []))
    leakage_cols = list(data_cfg.get("leakage_cols", []))
    sample_rows = None if args.rows and args.rows <= 0 else int(args.rows)
    generate_column_dictionary(
        csv_path=csv_path,
        parse_dates=parse_dates,
        leakage_cols=leakage_cols,
        out_path=Path(args.out),
        sample_rows=sample_rows,
        sparse_unknown_threshold=float(args.sparse_unknown_threshold),
    )


if __name__ == "__main__":
    main()
