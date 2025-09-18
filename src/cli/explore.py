from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.cli._bootstrap import apply_safe_env
from src.features.preprocess import identify_feature_types
from src.data.load import (
    compute_credit_history_length_months,
    _parse_date_series,  # type: ignore
)


def _load_config_with_extends(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    extends = cfg.get("extends")
    if extends:
        base_candidate = cfg_path.parent / f"{extends}.yaml"
        base_path = base_candidate if base_candidate.exists() else Path(extends)
        base_cfg = _load_config_with_extends(base_path)
        base_cfg.update({k: v for k, v in cfg.items() if k != "extends"})
        return base_cfg
    return cfg


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_pos_labels(y_raw: pd.Series, pos_label_cfg: int | str) -> np.ndarray:
    if isinstance(pos_label_cfg, str):
        pos_label_cfg = 0 if str(pos_label_cfg).lower() in {"default", "charged off", "charged_off"} else 1
    y = pd.to_numeric(y_raw, errors="coerce").astype("Int64").fillna(0).to_numpy()
    if int(pos_label_cfg) == 1:
        return y.astype(int)
    return (1 - y).astype(int)


def _year_quarter(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.to_period("Q").astype(str)


def _summarize_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []
    percentiles = [0.01, 0.05, 0.5, 0.95, 0.99]
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        desc = {
            "col": c,
            "count": int(s.notna().sum()),
            "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
            "std": float(s.std(skipna=True)) if s.notna().any() else np.nan,
            "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
            "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
        }
        try:
            qs = s.quantile(percentiles)
            desc.update({
                "p1": float(qs.get(0.01, np.nan)),
                "p5": float(qs.get(0.05, np.nan)),
                "p50": float(qs.get(0.5, np.nan)),
                "p95": float(qs.get(0.95, np.nan)),
                "p99": float(qs.get(0.99, np.nan)),
            })
        except Exception:
            desc.update({"p1": np.nan, "p5": np.nan, "p50": np.nan, "p95": np.nan, "p99": np.nan})
        out_rows.append(desc)
    return pd.DataFrame(out_rows)


def _summarize_categorical(df: pd.DataFrame, cols: List[str], top_k: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    card_rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    n = len(df)
    for c in cols:
        s = df[c]
        uniq = int(s.nunique(dropna=True))
        card_rows.append({"col": c, "unique": uniq})
        vc = s.value_counts(dropna=False).head(top_k)
        for k, v in vc.items():
            top_rows.append({"col": c, "level": (str(k) if not pd.isna(k) else "<NA>"), "count": int(v), "rate": float(v / max(n, 1))})
    return pd.DataFrame(card_rows), pd.DataFrame(top_rows)


def _compute_numeric_correlations(df: pd.DataFrame, num_cols: List[str], y_pos: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = y_pos.astype(float)
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 2:
            rows.append({"col": c, "corr": np.nan, "n": int(s.notna().sum())})
            continue
        m = s.median(skipna=True)
        x = s.fillna(m).to_numpy()
        try:
            r = float(np.corrcoef(x, y[: len(x)])[0, 1])
        except Exception:
            r = np.nan
        rows.append({"col": c, "corr": r, "n": int(len(x))})
    out = pd.DataFrame(rows)
    out["abs_corr"] = out["corr"].abs()
    out.sort_values("abs_corr", ascending=False, inplace=True)
    return out


def _compute_mutual_info_per_feature(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], y_pos: np.ndarray, sample_n: int, random_state: int) -> pd.DataFrame:
    from sklearn.feature_selection import mutual_info_classif

    rng = np.random.default_rng(random_state)
    if len(df) > sample_n:
        idx = rng.choice(len(df), size=sample_n, replace=False)
        df_s = df.iloc[idx].copy()
        y_s = y_pos[idx]
    else:
        df_s = df.copy()
        y_s = y_pos.copy()

    rows: List[Dict[str, Any]] = []
    # Numeric features (continuous)
    for c in num_cols:
        x = pd.to_numeric(df_s[c], errors="coerce").astype(float).to_numpy().reshape(-1, 1)
        x = np.nan_to_num(x, nan=np.nanmedian(x))
        try:
            mi = float(mutual_info_classif(x, y_s, discrete_features=False, random_state=random_state)[0])
        except Exception:
            mi = np.nan
        rows.append({"feature": c, "mi": mi, "type": "numeric"})

    # Categorical features (discrete via factorize)
    for c in cat_cols:
        codes, _ = pd.factorize(df_s[c], sort=False)
        codes = np.where(np.isnan(codes.astype(float)), -1, codes)  # ensure ints
        x = codes.reshape(-1, 1)
        try:
            mi = float(mutual_info_classif(x, y_s, discrete_features=True, random_state=random_state)[0])
        except Exception:
            mi = np.nan
        rows.append({"feature": c, "mi": mi, "type": "categorical"})

    out = pd.DataFrame(rows)
    out.sort_values("mi", ascending=False, inplace=True)
    return out


def _plot_top_numeric_histograms(df: pd.DataFrame, y_pos: np.ndarray, num_cols: List[str], mi_df: pd.DataFrame, fig_dir: Path, top_k: int = 12, fname_suffix: str = "") -> None:
    try:
        import matplotlib.pyplot as plt
        top = [f for f in mi_df.loc[mi_df["type"] == "numeric", "feature"].head(top_k).tolist() if f in num_cols]
        for c in top:
            s = pd.to_numeric(df[c], errors="coerce")
            pos = s[y_pos == 1].dropna()
            neg = s[y_pos == 0].dropna()
            if len(pos) == 0 and len(neg) == 0:
                continue
            # Shared bins via quantiles for better overlay
            all_s = pd.concat([pos, neg], axis=0)
            try:
                qs = np.nanpercentile(all_s.to_numpy(), [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
                bins = np.unique(qs)
                if len(bins) < 5:
                    bins = 30
            except Exception:
                bins = 30
            plt.figure(figsize=(8, 5))
            if len(neg):
                plt.hist(neg, bins=bins, alpha=0.5, label="negative", density=True, color="#1f77b4")
            if len(pos):
                plt.hist(pos, bins=bins, alpha=0.5, label="positive", density=True, color="#d62728")
            plt.title(f"Distribution by class: {c}")
            plt.legend()
            plt.tight_layout()
            suffix = ("_" + fname_suffix) if fname_suffix else ""
            plt.savefig(fig_dir / f"hist_{c}{suffix}.png")
            plt.close()
    except Exception:
        pass


def _plot_top_categorical_bars(df: pd.DataFrame, y_pos: np.ndarray, cat_cols: List[str], mi_df: pd.DataFrame, fig_dir: Path, top_k: int = 12, levels_k: int = 15, fname_suffix: str = "") -> None:
    try:
        import matplotlib.pyplot as plt
        top = [f for f in mi_df.loc[mi_df["type"] == "categorical", "feature"].head(top_k).tolist() if f in cat_cols]
        for c in top:
            s = df[c].astype("object")
            # Top levels overall
            vc = s.value_counts(dropna=False).head(levels_k)
            levels = vc.index.tolist()
            # Compute positive rate per level
            rates = []
            counts = []
            for lv in levels:
                mask = (s == lv) if not pd.isna(lv) else s.isna()
                n = int(mask.sum())
                counts.append(n)
                if n > 0:
                    pr = float(y_pos[mask.to_numpy()].mean())
                else:
                    pr = np.nan
                rates.append(pr)
            labels = ["<NA>" if pd.isna(lv) else str(lv) for lv in levels]
            fig, ax1 = plt.subplots(figsize=(max(8, min(16, 0.6 * len(labels) + 4)), 5))
            ax1.bar(range(len(labels)), counts, color="#1f77b4", alpha=0.7)
            ax1.set_ylabel("count")
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=45, ha="right")
            ax2 = ax1.twinx()
            ax2.plot(range(len(labels)), rates, color="#d62728", marker="o")
            ax2.set_ylabel("positive rate")
            plt.title(f"Counts and positive rate by level: {c}")
            fig.tight_layout()
            suffix = ("_" + fname_suffix) if fname_suffix else ""
            fig.savefig(fig_dir / f"cat_{c}{suffix}.png")
            plt.close(fig)
    except Exception:
        pass


def _compute_psi_numeric(df: pd.DataFrame, time_ser: pd.Series, num_cols: List[str], early_mask: pd.Series, late_mask: pd.Series, bins: int = 10) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    eps = 1e-6
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        ref = s[early_mask]
        cur = s[late_mask]
        try:
            qs = np.nanpercentile(ref.dropna().to_numpy(), np.linspace(0, 100, bins + 1))
            edges = np.unique(qs)
            if len(edges) < 3:
                raise ValueError("insufficient bins")
            ref_bin = pd.cut(ref, bins=edges, include_lowest=True)
            cur_bin = pd.cut(cur, bins=edges, include_lowest=True)
            ref_dist = ref_bin.value_counts(normalize=True, dropna=False).sort_index()
            cur_dist = cur_bin.value_counts(normalize=True, dropna=False).sort_index()
            # align indices
            all_idx = ref_dist.index.union(cur_dist.index)
            ref_p = ref_dist.reindex(all_idx).fillna(0).to_numpy() + eps
            cur_p = cur_dist.reindex(all_idx).fillna(0).to_numpy() + eps
            psi = float(np.sum((ref_p - cur_p) * np.log(ref_p / cur_p)))
            rows.append({"feature": c, "psi": psi, "n_early": int(ref.notna().sum()), "n_late": int(cur.notna().sum()), "bins": int(len(all_idx))})
        except Exception:
            rows.append({"feature": c, "psi": np.nan, "n_early": int(ref.notna().sum()), "n_late": int(cur.notna().sum()), "bins": 0})
    out = pd.DataFrame(rows)
    out.sort_values("psi", ascending=False, inplace=True)
    return out


def _compute_psi_categorical(df: pd.DataFrame, cat_cols: List[str], early_mask: pd.Series, late_mask: pd.Series, top_levels: int = 50) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    eps = 1e-6
    for c in cat_cols:
        s = df[c].astype("object")
        # limit to top levels globally to avoid huge cardinality
        levels = s.value_counts(dropna=False).head(top_levels).index
        ref = s[early_mask]
        cur = s[late_mask]
        ref_dist = ref.value_counts(normalize=True, dropna=False)
        cur_dist = cur.value_counts(normalize=True, dropna=False)
        # align on chosen levels only
        idx = pd.Index(levels)
        ref_p = ref_dist.reindex(idx).fillna(0).to_numpy() + eps
        cur_p = cur_dist.reindex(idx).fillna(0).to_numpy() + eps
        psi = float(np.sum((ref_p - cur_p) * np.log(ref_p / cur_p)))
        rows.append({"feature": c, "psi": psi, "levels": int(len(idx))})
    out = pd.DataFrame(rows)
    out.sort_values("psi", ascending=False, inplace=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore dataset: class balance, missingness, distributions, MI")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV override (defaults to config data.csv_path)")
    args = parser.parse_args()

    # Safe environment for headless/BLAS
    apply_safe_env()

    cfg = _load_config_with_extends(Path(args.config))
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    split_cfg = cfg.get("split", {})

    csv_path = Path(args.csv or data_cfg.get("csv_path", "")).resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    # Build usecols: features + target + parse_dates + leakage (for inspection)
    features: List[str] = list(data_cfg.get("features", []))
    parse_dates_cols: List[str] = list(data_cfg.get("parse_dates", []))
    target_col: str = data_cfg.get("target_col")
    leakage_cols: List[str] = list(data_cfg.get("leakage_cols", []))
    usecols_list = list(dict.fromkeys(features + parse_dates_cols + [target_col] + leakage_cols))

    # Read CSV once (no chunking); parse dates post-read using project-specific parser
    t0 = time.time()
    df = pd.read_csv(csv_path, usecols=lambda c: (c in usecols_list), low_memory=False)
    # Target mapping and filter
    tmap: Dict[str, int] = dict(data_cfg.get("target_mapping", {}))
    if target_col not in df.columns:
        raise SystemExit(f"Target column '{target_col}' missing in dataset")
    df = df[df[target_col].isin(tmap.keys())].copy()
    df[target_col] = df[target_col].map(tmap)

    # Parse dates with repository parser for known formats
    for col in parse_dates_cols:
        if col in df.columns:
            df[col] = _parse_date_series(df[col])

    # Engineered features for analysis
    if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
        df["credit_history_length"] = compute_credit_history_length_months(df["issue_d"], df["earliest_cr_line"])
    if "annual_inc" in df.columns and "loan_amnt" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["income_to_loan_ratio"] = (pd.to_numeric(df["annual_inc"], errors="coerce") / pd.to_numeric(df["loan_amnt"], errors="coerce")).replace([np.inf, -np.inf], np.nan)
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        low = pd.to_numeric(df["fico_range_low"], errors="coerce")
        high = pd.to_numeric(df["fico_range_high"], errors="coerce")
        df["fico_avg"] = (low + high) / 2.0
        df["fico_spread"] = high - low

    # Identify feature types
    num_cols, cat_cols = identify_feature_types(df.drop(columns=[target_col], errors="ignore"))
    # Avoid date/datetime columns in numeric list
    for dcol in parse_dates_cols:
        if dcol in num_cols:
            num_cols.remove(dcol)

    # Origination-only view: drop known leakage columns
    leak_present = [c for c in leakage_cols if c in df.columns]
    df_orig = df.drop(columns=leak_present, errors="ignore") if leak_present else df.copy()
    num_cols_orig, cat_cols_orig = identify_feature_types(df_orig.drop(columns=[target_col], errors="ignore"))
    for dcol in parse_dates_cols:
        if dcol in num_cols_orig:
            num_cols_orig.remove(dcol)

    # Memory stats
    mem_bytes = int(df.memory_usage(deep=True).sum())

    # Positive class alignment
    pos_label_cfg = eval_cfg.get("pos_label", 1)
    y_pos = _as_pos_labels(df[target_col], pos_label_cfg)

    # Class balance overall
    n = int(len(df))
    n_pos = int(y_pos.sum())
    n_neg = int(n - n_pos)
    pos_rate = float(n_pos / max(n, 1))

    # By time (year / quarter)
    cls_by_year = pd.DataFrame()
    cls_by_quarter = pd.DataFrame()
    if "issue_d" in df.columns:
        year = pd.to_datetime(df["issue_d"], errors="coerce").dt.year
        grp = pd.DataFrame({"year": year, "pos": y_pos}).dropna()
        agg = grp.groupby("year").agg(n=("pos", "size"), n_pos=("pos", "sum"))
        agg["pos_rate"] = agg["n_pos"] / agg["n"].clip(lower=1)
        cls_by_year = agg.reset_index().astype({"year": int})

        q = _year_quarter(df["issue_d"])
        grpq = pd.DataFrame({"quarter": q, "pos": y_pos})
        aggq = grpq.groupby("quarter").agg(n=("pos", "size"), n_pos=("pos", "sum"))
        aggq["pos_rate"] = aggq["n_pos"] / aggq["n"].clip(lower=1)
        cls_by_quarter = aggq.reset_index()

    # Missingness
    miss = df.isna().sum().to_frame(name="missing").reset_index().rename(columns={"index": "col"})
    miss["rate"] = miss["missing"] / max(n, 1)
    miss.sort_values("rate", ascending=False, inplace=True)

    # Categorical summaries
    cat_card, cat_top = _summarize_categorical(df, cat_cols)

    # Numeric summaries
    num_summary = _summarize_numeric(df, num_cols)

    # Numeric correlations vs configured positive class
    corr_num = _compute_numeric_correlations(df, num_cols, y_pos)

    # Mutual information (sampled heavy calc)
    rs = int(split_cfg.get("random_state", 42))
    mi_df = _compute_mutual_info_per_feature(df, num_cols, cat_cols, y_pos, sample_n=200_000, random_state=rs)

    # Origination-only correlations and MI
    corr_num_orig = _compute_numeric_correlations(df_orig, num_cols_orig, y_pos)
    mi_df_orig = _compute_mutual_info_per_feature(df_orig, num_cols_orig, cat_cols_orig, y_pos, sample_n=200_000, random_state=rs)

    # Drift-ish summaries (per-year means for top movers by std across years)
    drift_rows: List[Dict[str, Any]] = []
    if "issue_d" in df.columns and not cls_by_year.empty:
        df_year = pd.to_datetime(df["issue_d"], errors="coerce").dt.year
        for c in num_cols[:]:
            s = pd.to_numeric(df[c], errors="coerce")
            tmp = pd.DataFrame({"year": df_year, c: s})
            g = tmp.dropna().groupby("year")[c].mean()
            if len(g) >= 2:
                years = g.index.values.astype(float)
                vals = g.values.astype(float)
                try:
                    slope = float(np.polyfit(years, vals, 1)[0])
                except Exception:
                    slope = np.nan
                drift_rows.append({"col": c, "years": int(len(g)), "std_over_year_means": float(np.nanstd(vals)), "slope_vs_year": slope})
    drift_df = pd.DataFrame(drift_rows).sort_values("std_over_year_means", ascending=False) if drift_rows else pd.DataFrame(columns=["col", "years", "std_over_year_means", "slope_vs_year"])

    # Output folder
    run_id = time.strftime("run_%Y%m%d_%H%M%S_explore")
    run_dir = Path("local_runs") / run_id
    fig_dir = run_dir / "figures"
    _ensure_dir(fig_dir)

    # Save tables
    def _save_df(d: pd.DataFrame, name: str) -> None:
        d.to_csv(run_dir / name, index=False)

    _save_df(pd.DataFrame([{"total": n, "n_positive": n_pos, "n_negative": n_neg, "pos_rate": pos_rate}]), "class_balance_overall.csv")
    if not cls_by_year.empty:
        _save_df(cls_by_year, "class_balance_by_year.csv")
    if not cls_by_quarter.empty:
        _save_df(cls_by_quarter, "class_balance_by_quarter.csv")
    _save_df(miss, "missingness.csv")
    _save_df(cat_card, "categorical_cardinality.csv")
    _save_df(cat_top, "categorical_top_levels.csv")
    _save_df(num_summary, "numeric_summary.csv")
    _save_df(corr_num, "feature_corr_numeric.csv")
    _save_df(mi_df, "feature_mi.csv")
    _save_df(corr_num_orig, "feature_corr_numeric_orig.csv")
    _save_df(mi_df_orig, "feature_mi_orig.csv")
    _save_df(drift_df, "drift_feature_summaries.csv")

    # Figures
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    try:
        if not cls_by_year.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(cls_by_year["year"], cls_by_year["pos_rate"], marker="o")
            plt.title("Positive rate over years (configured positive class)")
            plt.xlabel("Year")
            plt.ylabel("Positive rate")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / "class_balance_over_time.png")
            plt.close()
    except Exception:
        pass

    try:
        top_miss = miss.head(30)
        plt.figure(figsize=(10, 8))
        plt.barh(top_miss["col"][::-1], top_miss["rate"][::-1])
        plt.title("Top missingness (rate)")
        plt.xlabel("Missing rate")
        plt.tight_layout()
        plt.savefig(fig_dir / "missingness_top.png")
        plt.close()
    except Exception:
        pass

    # Bar chart: top numeric correlations
    try:
        top_corr = corr_num.dropna(subset=["abs_corr"]).head(30)
        plt.figure(figsize=(10, 8))
        plt.barh(top_corr["col"][::-1], top_corr["abs_corr"][::-1])
        plt.title("Top |correlation| with target (numeric)")
        plt.xlabel("|corr|")
        plt.tight_layout()
        plt.savefig(fig_dir / "top_corr_numeric.png")
        plt.close()
    except Exception:
        pass

    # Bar chart: top numeric correlations (origination-only)
    try:
        top_corr_o = corr_num_orig.dropna(subset=["abs_corr"]).head(30)
        plt.figure(figsize=(10, 8))
        plt.barh(top_corr_o["col"][::-1], top_corr_o["abs_corr"][::-1])
        plt.title("Top |correlation| with target (numeric) — orig-only")
        plt.xlabel("|corr|")
        plt.tight_layout()
        plt.savefig(fig_dir / "top_corr_numeric_orig.png")
        plt.close()
    except Exception:
        pass

    # Histograms for top MI numeric and bar charts for top MI categorical
    _plot_top_numeric_histograms(df, y_pos, num_cols, mi_df, fig_dir, top_k=12)
    _plot_top_categorical_bars(df, y_pos, cat_cols, mi_df, fig_dir, top_k=12, levels_k=15)
    # Origination-only counterparts
    _plot_top_numeric_histograms(df_orig, y_pos, num_cols_orig, mi_df_orig, fig_dir, top_k=12, fname_suffix="orig")
    _plot_top_categorical_bars(df_orig, y_pos, cat_cols_orig, mi_df_orig, fig_dir, top_k=12, levels_k=15, fname_suffix="orig")

    # PSI early vs late using configured time column
    psi_meta: Dict[str, Any] = {}
    try:
        time_col = split_cfg.get("time_col", "issue_d")
        if time_col in df.columns:
            s_time = pd.to_datetime(df[time_col], errors="coerce")
            # Use config test_size to define late period
            test_size = float(split_cfg.get("test_size", 0.2))
            q = 1.0 - max(0.0, min(1.0, test_size))
            cut_date = s_time.quantile(q)
            early_mask = s_time <= cut_date
            late_mask = s_time > cut_date
            psi_meta = {"time_col": time_col, "cut_date": (None if pd.isna(cut_date) else str(pd.Timestamp(cut_date).date())), "early_n": int(early_mask.sum()), "late_n": int(late_mask.sum())}
            psi_num = _compute_psi_numeric(df, s_time, num_cols, early_mask, late_mask, bins=10)
            psi_cat = _compute_psi_categorical(df, cat_cols, early_mask, late_mask, top_levels=50)
            _save_df(psi_num, "psi_numeric.csv")
            _save_df(psi_cat, "psi_categorical.csv")
            # Origination-only PSI
            psi_num_orig = _compute_psi_numeric(df_orig, s_time, num_cols_orig, early_mask, late_mask, bins=10)
            psi_cat_orig = _compute_psi_categorical(df_orig, cat_cols_orig, early_mask, late_mask, top_levels=50)
            _save_df(psi_num_orig, "psi_numeric_orig.csv")
            _save_df(psi_cat_orig, "psi_categorical_orig.csv")
            # Plot top PSI bars
            try:
                topn = psi_num.head(20)
                plt.figure(figsize=(10, 8))
                plt.barh(topn["feature"][::-1], topn["psi"][::-1])
                plt.title("Top PSI (numeric)")
                plt.xlabel("PSI")
                plt.tight_layout()
                plt.savefig(fig_dir / "psi_numeric_top.png")
                plt.close()
            except Exception:
                pass
            try:
                topc = psi_cat.head(20)
                plt.figure(figsize=(10, 8))
                plt.barh(topc["feature"][::-1], topc["psi"][::-1])
                plt.title("Top PSI (categorical)")
                plt.xlabel("PSI")
                plt.tight_layout()
                plt.savefig(fig_dir / "psi_categorical_top.png")
                plt.close()
            except Exception:
                pass
            # Origination-only PSI plots
            try:
                topn_o = psi_num_orig.head(20)
                plt.figure(figsize=(10, 8))
                plt.barh(topn_o["feature"][::-1], topn_o["psi"][::-1])
                plt.title("Top PSI (numeric) — orig-only")
                plt.xlabel("PSI")
                plt.tight_layout()
                plt.savefig(fig_dir / "psi_numeric_top_orig.png")
                plt.close()
            except Exception:
                pass
            try:
                topc_o = psi_cat_orig.head(20)
                plt.figure(figsize=(10, 8))
                plt.barh(topc_o["feature"][::-1], topc_o["psi"][::-1])
                plt.title("Top PSI (categorical) — orig-only")
                plt.xlabel("PSI")
                plt.tight_layout()
                plt.savefig(fig_dir / "psi_categorical_top_orig.png")
                plt.close()
            except Exception:
                pass
    except Exception:
        pass

    # Summary JSON
    summary: Dict[str, Any] = {
        "csv_path": csv_path.as_posix(),
        "rows": n,
        "memory_bytes": mem_bytes,
        "memory_mb": round(mem_bytes / (1024 * 1024), 2),
        "pos_label": int(0 if (isinstance(pos_label_cfg, str) and str(pos_label_cfg).lower() in {"default", "charged off", "charged_off"}) else pos_label_cfg),
        "pos_rate": pos_rate,
        "n_numeric": int(len(num_cols)),
        "n_categorical": int(len(cat_cols)),
        "mi_sample_rows": int(min(200_000, n)),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    if psi_meta:
        summary["psi"] = psi_meta
    try:
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(run_dir / "features.json", "w", encoding="utf-8") as f:
            json.dump({
                "numeric": num_cols,
                "categorical": cat_cols,
                "numeric_orig_only": num_cols_orig,
                "categorical_orig_only": cat_cols_orig,
                "leakage_cols_present": leak_present,
            }, f, indent=2)
    except Exception:
        pass

    # Console pointer
    print(json.dumps({"run_dir": run_dir.as_posix(), **summary}, indent=2))


if __name__ == "__main__":
    main()
