from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_catalog(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|")


def _run_row(run: Dict[str, Any]) -> str:
    rid = run.get("run_id", "")
    backend = run.get("backend") or "?"
    metrics = run.get("metrics") or {}
    auc = metrics.get("auc") or metrics.get("val_auc") or metrics.get("roc_auc")
    auc_str = f"{auc:.3f}" if isinstance(auc, (int, float)) else "-"
    thr = metrics.get("threshold") or "-"
    path = run.get("path", "")
    figures = run.get("figures") or []
    fig_link = f"[figures]({_escape_md(Path(path).name)}/figures)" if figures else "-"
    return f"{_escape_md(rid)} | {_escape_md(backend)} | {auc_str} | {thr} | [{_escape_md(Path(path).name)}]({_escape_md(Path(path).name)}) | {fig_link}"


def build_markdown(catalog: Dict[str, Any]) -> str:
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
        lines.append("run_id | backend | AUC | thr | folder | figures")
        lines.append(":-- | :-- | --: | :--: | :-- | :--")
        for r in sorted(rs, key=lambda r: r.get("run_id", "")):
            lines.append(_run_row(r))
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
    md = build_markdown(catalog)

    out_path = Path(args.out) if args.out else runs_root / "index.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(json.dumps({"wrote": out_path.as_posix(), "runs": int(catalog.get("count", 0))}, indent=2))


if __name__ == "__main__":
    main()

