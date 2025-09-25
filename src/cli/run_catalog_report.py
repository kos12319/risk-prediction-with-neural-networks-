from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import math

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None  # type: ignore


def _load_catalog(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|")


def _run_row(run: Dict[str, Any], delta_auc: str | None = None) -> str:
    rid = run.get("run_id", "")
    backend = run.get("backend") or "?"
    metrics = run.get("metrics") or {}
    auc = metrics.get("auc") or metrics.get("val_auc") or metrics.get("roc_auc")
    auc_str = f"{auc:.3f}" if isinstance(auc, (int, float)) else "-"
    if delta_auc is None:
        delta_auc = "-"
    thr = metrics.get("threshold") or "-"
    path = run.get("path", "")
    figures = run.get("figures") or []
    fig_link = f"[figures]({_escape_md(Path(path).name)}/figures)" if figures else "-"
    return (
        f"{_escape_md(rid)} | {_escape_md(backend)} | {auc_str} | {delta_auc} | {thr} | "
        f"[{_escape_md(Path(path).name)}]({_escape_md(Path(path).name)}) | {fig_link}"
    )


def _group_key(run: Dict[str, Any]) -> Tuple:
    # Prefer created timestamp (float str), fallback to run_id lexical
    created = run.get("created")
    try:
        cval = float(created) if created is not None else math.inf
    except Exception:
        cval = math.inf
    return (cval, str(run.get("run_id", "")))


def build_markdown(catalog: Dict[str, Any], *, runs_root: Path, make_plots: bool = True) -> str:
    runs: List[Dict[str, Any]] = list(catalog.get("runs") or [])
    # Group by parent folder (group)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        g = r.get("group") or "ungrouped"
        groups.setdefault(g, []).append(r)

    lines: List[str] = []
    lines.append("# Run Catalog")
    lines.append("")
    lines.append(f"Root: `{catalog.get('runs_root', '')}`  ")
    lines.append(f"Runs: {int(catalog.get('count', 0))}")
    lines.append("")

    for g, rs in sorted(groups.items(), key=lambda kv: kv[0]):
        lines.append(f"## {_escape_md(g)}")
        lines.append("")
        # Optionally emit a simple AUC trend plot per group
        plot_rel = None
        if make_plots and plt is not None:
            try:
                # Sort runs by created time
                rs_sorted = sorted(rs, key=_group_key)
                xs = list(range(1, len(rs_sorted) + 1))
                ys = [
                    (r.get("metrics") or {}).get("auc")
                    or (r.get("metrics") or {}).get("val_auc")
                    or (r.get("metrics") or {}).get("roc_auc")
                    for r in rs_sorted
                ]
                ys_float = [float(y) for y in ys if isinstance(y, (int, float))]
                if ys_float:
                    figdir = runs_root / "index_plots"
                    figdir.mkdir(parents=True, exist_ok=True)
                    slug = g.replace("/", "_").replace(" ", "_")
                    out_png = figdir / f"trend_auc_{slug}.png"
                    plt.figure(figsize=(4.0, 2.2), dpi=150)
                    plt.plot(xs, [float(y) if isinstance(y, (int, float)) else float('nan') for y in ys], marker="o")
                    plt.title(f"AUC trend — {g}")
                    plt.xlabel("run")
                    plt.ylabel("AUC")
                    plt.ylim(0.0, 1.0)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(out_png.as_posix())
                    plt.close()
                    plot_rel = out_png.relative_to(runs_root).as_posix()
            except Exception:
                plot_rel = None

        if plot_rel:
            lines.append(f"Trend: ![]({_escape_md(plot_rel)})")
            lines.append("")

        lines.append("run_id | backend | AUC | ΔAUC | thr | folder | figures")
        lines.append(":-- | :-- | --: | --: | :--: | :-- | :--")

        prev_auc: float | None = None
        for r in sorted(rs, key=_group_key):
            metrics = r.get("metrics") or {}
            auc = metrics.get("auc") or metrics.get("val_auc") or metrics.get("roc_auc")
            delta_s = "-"
            if isinstance(auc, (int, float)) and isinstance(prev_auc, (int, float)):
                delta = float(auc) - float(prev_auc)
                delta_s = f"{delta:+.3f}"
            lines.append(_run_row(r, delta_s))
            prev_auc = float(auc) if isinstance(auc, (int, float)) else prev_auc
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown report from _catalog.json")
    parser.add_argument("--runs-root", type=str, default="local_runs", help="Root folder for runs")
    parser.add_argument("--catalog", type=str, default=None, help="Path to _catalog.json (defaults to <runs-root>/_catalog.json)")
    parser.add_argument("--out", type=str, default=None, help="Output Markdown (defaults to <runs-root>/index.md)")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    cat_path = Path(args.catalog) if args.catalog else runs_root / "_catalog.json"
    if not cat_path.exists():
        raise SystemExit(f"Catalog not found: {cat_path}")

    catalog = _load_catalog(cat_path)
    md = build_markdown(catalog, runs_root=runs_root, make_plots=True)

    out_path = Path(args.out) if args.out else runs_root / "index.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(json.dumps({"wrote": out_path.as_posix(), "runs": int(catalog.get("count", 0))}, indent=2))


if __name__ == "__main__":
    main()
