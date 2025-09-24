# Change & Decision Journal — How to Write Entries

Use this folder to chronicle meaningful platform changes. Each entry is small, dated, and links to relevant ADRs/configs.

Create via Makefile (preferred)
- `make docs-journal-new TITLE="<short title>" [TAGS="data,eval"] [ADRS="0001,0004"]`

Entry template (fields)
- Title: short, action-oriented
- Date: auto-filled by the Makefile helper
- Status: planned | in-progress | landed | reverted
- Summary: 2–4 lines describing the change and motivation
- ADRs: list of related ADR IDs (link to docs/ADRs)
- Impact: bullets (components, configs, Make targets)
- Next: optional follow-ups

Example
```
# Switch to temporal CV for selection

- Date: 2025-09-24
- Status: landed
- Tags: eval, cv

## Summary
Adopt forward-chaining temporal CV for feature selection to stabilize subsets across vintages.

## ADRs
- 0002 — temporal CV for selection (proposed)
- 0001 — time-based split for evaluation (accepted)

## Impact
- configs: `split.cv: expanding`, `train_full_after: true`
- Make: `make train` now logs `reports/cv_metrics.json`
- code: `src/data/split.py`, `src/selection/`

## Next
- Evaluate k settings for expanding window; update ADR 0002 to accepted.
```

