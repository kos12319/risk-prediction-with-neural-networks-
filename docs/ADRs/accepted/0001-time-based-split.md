# ADR 0001 — Use Time-Based Split for Model Evaluation

- Status: Accepted
- Date: 2025-09-16

## Context
Credit risk at origination is inherently time-ordered. Cohorts of loans reflect macro conditions, underwriting policies, and borrower mix that shift over time (concept drift). Evaluating models with random splits risks mixing future information into training, overstating performance.

## Decision
Adopt a time-based holdout split by `issue_d` as the default evaluation strategy:
- Train on earlier loans; test on the most recent portion (e.g., last 20%).
- Configuration controls are `split.method: time` and `split.time_col: issue_d`.

## Rationale
- Mirrors production reality: you train on past loans and must predict future outcomes; random splits leak “future” patterns into training and overstate performance.
- Avoids temporal leakage: macro cycles, underwriting policy changes, and borrower mix shift over time; a time split reflects this distribution shift instead of mixing cohorts.
- Domain fit: credit risk at origination is inherently time‑ordered; using past→future evaluation yields a conservative, realistic ROC AUC and PR.
- Implementation: default config sets `split.method: time` with `time_col: issue_d` (configs/default.yaml:94), and training calls the time split when configured (src/data/split.py, src/training/train_nn.py).
- Engineered dates: features like `credit_history_length` depend on `issue_d`; a time split ensures those are computed and evaluated in a forward‑looking way.
- Leakage controls complement: project already drops post‑origination fields; time split is the second guardrail against optimistic bias.
- When to switch: for quick debugging or small, stationary samples, set `split.method: random` — expect higher but less reliable metrics.
- Next step (optional): for model/feature selection, consider forward‑chaining CV (rolling windows) rather than a single holdout to stabilize estimates.

## Consequences
- Reported metrics are typically lower than random splits but more faithful to deployment performance.
- CI and experimentation should use comparable time windows to avoid cohort effects.
- Feature selection and hyperparameter tuning should prefer temporal CV to reduce variance.
- Data loading must ensure `parse_dates` columns (e.g., `issue_d`) are present to avoid silent failures.

## Alternatives Considered
- Random stratified split: faster and higher apparent performance but risks temporal leakage and optimistic bias.
- K‑fold CV without time awareness: mixes future/past folds; not appropriate for temporal data.
- Blocked/rolling temporal CV: more robust than single holdout; chosen later as an enhancement due to added complexity/compute.

## Implementation Notes
- Config: `configs/default.yaml` uses `split.method: time` and `split.time_col: issue_d`.
- Code: `src/data/split.py` provides `time_based_split`; `src/training/train_nn.py` invokes it when configured.
- Plots and metrics reflect the chosen positive class and threshold strategy.

