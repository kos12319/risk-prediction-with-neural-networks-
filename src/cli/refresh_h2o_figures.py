from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def _shorten_model_id(model_id: str) -> str:
    if not isinstance(model_id, str):
        return str(model_id)
    if "_AutoML" in model_id:
        return model_id.split("_AutoML", 1)[0]
    return model_id


def _build_label_map(*frames: Optional[pd.DataFrame]) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    seen_labels: Dict[str, int] = {}
    for frame in frames:
        if frame is None or "model_id" not in frame.columns:
            continue
        reset = frame.reset_index(drop=True)
        for rank, row in reset.iterrows():
            model_id = row.get("model_id")
            if not isinstance(model_id, str):
                model_id = str(model_id)
            if not model_id or model_id in label_map:
                continue
            algo = None
            for key in ("algo", "model_type", "model_category"):
                if key in row and pd.notna(row[key]):
                    algo = str(row[key]).strip()
                    break
            short_id = _shorten_model_id(model_id)
            base_label = algo if algo else short_id
            label = f"{rank + 1}. {base_label}"
            if short_id not in base_label:
                label = f"{label} [{short_id}]"
            if label in seen_labels:
                seen_labels[label] += 1
                label = f"{label} #{seen_labels[label]}"
            else:
                seen_labels[label] = 1
            label_map[model_id] = label
    return label_map


def _plot_best_per_category(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    lb_df = df.copy()
    label_map = _build_label_map(lb_df)
    # Try to identify a category column
    cat_col = None
    for c in ("model_category", "algo", "model_type"):
        if c in lb_df.columns:
            cat_col = c
            break
    if cat_col is None or "auc" not in lb_df.columns or "model_id" not in lb_df.columns:
        return
    try:
        lb_df["auc"] = pd.to_numeric(lb_df["auc"], errors="coerce")
    except Exception:
        pass
    winners = lb_df[["model_id", cat_col, "auc"]].dropna().sort_values("auc", ascending=False).groupby(cat_col, as_index=False).first()
    # Reindex with display labels for legend clarity
    winners.index = [label_map.get(str(mid), str(mid)) for mid in winners["model_id"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    winners.set_index(cat_col)["auc"].sort_values(ascending=False).plot(kind="barh", ax=ax, color="#2ca02c")
    ax.invert_yaxis()
    ax.set_xlabel("AUC")
    ax.set_ylabel("Category")
    ax.set_title("H2O — Best Model per Category (AUC)")
    plt.tight_layout()
    fig.savefig(outdir / "h2o_best_per_category_auc.png", dpi=150)
    plt.close(fig)


def _plot_family_leaderboards(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    lb_df = df.copy()
    # Identify family column
    fam_col = None
    for c in ("model_category", "algo", "model_type"):
        if c in lb_df.columns:
            fam_col = c
            break
    if fam_col is None or "model_id" not in lb_df.columns:
        return
    def plot_metric(metric: str, higher_is_better: bool, title: str) -> None:
        if metric not in lb_df.columns:
            return
        tmp = lb_df[["model_id", fam_col, metric]].dropna().copy()
        try:
            tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        except Exception:
            pass
        tmp = tmp.dropna(subset=[metric])
        if tmp.empty:
            return
        winners = tmp.sort_values(metric, ascending=not higher_is_better).groupby(fam_col, as_index=False).first()
        series = winners.set_index(fam_col)[metric].sort_values(ascending=not higher_is_better)
        fig, ax = plt.subplots(figsize=(10, 6))
        direction_txt = "higher is better" if higher_is_better else "lower is better"
        arrow = "↑" if higher_is_better else "↓"
        pretty_metric = "Average Precision (AP)" if metric == "aucpr" else metric.upper()
        series.name = f"{pretty_metric} ({direction_txt} {arrow})"
        series.plot(kind="barh", ax=ax, color="#1f77b4", legend=True)
        ax.invert_yaxis()
        ax.set_xlabel(pretty_metric)
        ax.set_ylabel("Category")
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        fig.savefig(outdir / f"h2o_leaderboard_{metric}.png", dpi=150)
        plt.close(fig)

    plot_metric("aucpr", True, "H2O Leaderboard — Average Precision (by family)")
    plot_metric("logloss", False, "H2O Leaderboard — Log Loss (by family)")
    plot_metric("rmse", False, "H2O Leaderboard — RMSE (by family)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate H2O leaderboard figures with legends")
    parser.add_argument("--run-dir", required=True, type=Path, help="Path to a run directory containing h2o_leaderboard.csv")
    args = parser.parse_args()
    run_dir: Path = args.run_dir
    csv_path = run_dir / "h2o_leaderboard.csv"
    if not csv_path.exists():
        raise SystemExit(f"Leaderboard CSV not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise SystemExit(f"Failed to read leaderboard CSV: {exc}")
    figures_dir = run_dir / "figures"
    _plot_best_per_category(df, figures_dir)
    _plot_family_leaderboards(df, figures_dir)
    print(f"Refreshed family/category figures in: {figures_dir}")


if __name__ == "__main__":
    main()
