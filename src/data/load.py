from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class LoadConfig:
    csv_path: str
    target_col: str
    target_mapping: Dict[str, int]
    parse_dates: Sequence[str]
    drop_leakage: bool
    leakage_cols: Sequence[str]
    features: Sequence[str]


DATE_FORMATS = [
    "%b-%Y",  # e.g., Dec-2015
    "%b-%y",   # e.g., Dec-15
]


def _parse_date_series(s: pd.Series) -> pd.Series:
    # Try provided formats first, then fall back to pandas inference
    out = None
    for fmt in DATE_FORMATS:
        try:
            out = pd.to_datetime(s, format=fmt, errors="coerce")
            if out.notna().any():
                break
        except Exception:
            out = None
    if out is None or out.isna().all():
        out = pd.to_datetime(s, errors="coerce")
    return out


def compute_credit_history_length_months(issue_d: pd.Series, earliest_cr_line: pd.Series) -> pd.Series:
    issue_d_parsed = _parse_date_series(issue_d)
    earliest_parsed = _parse_date_series(earliest_cr_line)
    delta = issue_d_parsed - earliest_parsed
    # Convert to months (approximate by 30 days)
    months = (delta.dt.days / 30.0).round().astype("Int64")
    return months


def load_and_prepare(cfg: LoadConfig) -> pd.DataFrame:
    # Ensure we request the declared features plus target/date cols if not present
    usecols = list(dict.fromkeys(cfg.features + [cfg.target_col]))

    # Read
    df = pd.read_csv(cfg.csv_path, usecols=lambda c: (c in usecols), low_memory=False)

    # Target filter and map
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in dataset.")

    # Keep only rows where target in mapping keys
    valid_targets = set(cfg.target_mapping.keys())
    df = df[df[cfg.target_col].isin(valid_targets)].copy()
    df[cfg.target_col] = df[cfg.target_col].map(cfg.target_mapping)

    # Parse date columns if present
    for col in cfg.parse_dates:
        if col in df.columns:
            df[col] = _parse_date_series(df[col])

    # Engineer features when possible
    # credit_history_length (months)
    if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
        df["credit_history_length"] = compute_credit_history_length_months(
            df["issue_d"], df["earliest_cr_line"]
        )
    # income_to_loan_ratio
    if "annual_inc" in df.columns and "loan_amnt" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["income_to_loan_ratio"] = (df["annual_inc"] / df["loan_amnt"]).replace([np.inf, -np.inf], np.nan)
    # fico_avg and fico_spread
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_avg"] = (pd.to_numeric(df["fico_range_low"], errors="coerce") + pd.to_numeric(df["fico_range_high"], errors="coerce")) / 2.0
        df["fico_spread"] = pd.to_numeric(df["fico_range_high"], errors="coerce") - pd.to_numeric(df["fico_range_low"], errors="coerce")

    # Drop leakage columns if configured and present
    if cfg.drop_leakage:
        drops = [c for c in cfg.leakage_cols if c in df.columns]
        if drops:
            df = df.drop(columns=drops)

    return df.reset_index(drop=True)
