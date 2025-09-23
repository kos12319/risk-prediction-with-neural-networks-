# Thesis Proposal — Credit Risk Modeling Platform

This document reframes the project as a rigorous, MSc‑level thesis platform. It consolidates findings from the current codebase and exploration docs, proposes a clear thesis scope and hypotheses, and defines the methodology, experiments, and engineering work needed for reproducible, defensible results.

## Executive Summary
- Thesis focus: portability and robustness of neural credit‑risk models under temporal and provider shifts, guided by stability‑driven feature selection, calibration, and utility‑aligned thresholding.
- Two evaluation regimes: provider‑agnostic (excludes pricing/scoring fields) vs provider‑aware (includes `int_rate`, `grade`, `sub_grade`).
- Core deliverables:
  1. Time‑aware stability selection yielding compact, robust feature sets with near‑maximal AUC.
  2. Calibration analysis across vintages with mitigation via post‑hoc calibration.
  3. Utility‑aligned threshold selection and partial ROC in business‑relevant FPR ranges.
  4. Comparative study of deep tabular models vs gradient boosting under both regimes.

Dataset note: LendingClub consumer installment loans (2007–2018). The accepted-loans file contains funded applications with final outcomes; rejected-loans file has limited covariates for declined applications. Labels are derived from final statuses (Charged Off vs Fully Paid) with attention to right‑censoring. Features used for prediction remain strictly origination-time; all post-event fields (payments, recoveries, last_* dates, hardship/settlement) are excluded end-to-end.

## Research Questions & Hypotheses
- **RQ1**: How does excluding provider-specific fields affect out-of-time generalization and calibration?
  - **H1**: Provider-aware models score higher in-distribution but exhibit larger calibration drift; provider-agnostic models generalize more consistently.
- **RQ2**: Can a time-stable subset (15–30 features) maintain ≥95% of full-feature AUC while reducing drift sensitivity?
  - **H2**: Temporal stability selection (MI+L1 ensemble across forward-chaining folds) yields compact subsets with minimal AUC loss, improved calibration stability, and lower PSI sensitivity.
- **RQ3**: How do BCE vs focal loss behave in calibration, and can post-hoc methods recover calibration?
  - **H3**: Focal boosts recall but harms calibration; Platt/Isotonic/Temperature scaling restore Brier/ECE to competitive levels.
- **RQ4**: Do utility-based thresholds outperform generic choices (Youden/F1) under base-rate shift?
  - **H4**: Thresholds optimized for expected utility on validation maintain superior expected value on unseen test sets, especially when prevalence shifts.
- **RQ5**: Under what conditions do deep tabular networks match GBDTs on this task?
  - **H5**: MLPs with residual blocks and categorical embeddings close the gap to gradient boosting, particularly with provider-aware features; embeddings are crucial in the agnostic setting.

## Methodology & Invariants
- **Splitting**: Time-based train/test by `issue_d`; carve validation from the training period only. Oversampling, if any, applies to the training subset; the validation/test splits remain untouched.
- **Reproducibility**: Seed Python/NumPy/Torch/DataLoader workers; respect `eval.pos_label` (0 = Charged Off).
- **Preprocessing**: Same pipeline as training (impute, winsorize if enabled, scale numerics; impute + one-hot categoricals). Fit on train, apply to val/test.
- **Hyperparameter tuning**: Forward-chaining temporal CV (e.g., [2007–2014]→2015, [2007–2015]→2016, etc.); fix best hyperparameters; retrain on full train; evaluate once on held-out test.
- **Model families**: logistic (L1/L2), Random Forest, XGBoost/LightGBM/CatBoost; MLPs (baseline, residual, embeddings), optional FT-Transformer or TabNet for breadth.
- **Calibration/thresholding**: Evaluate Platt, Isotonic, Temperature scaling on the validation split; apply calibrated model to test. Compare fixed 0.5, Youden J, F1, and utility-based thresholds (with explicit cost matrix).
- **Metrics**: ROC-AUC (with DeLong CIs), PR-AUC/AP (bootstrap CIs), Brier score, ECE/MCE, confusion metrics at the chosen threshold, expected utility, PSI/CSI for drift. Where applicable, run statistical significance tests.

## Experiment Roadmap
1. **Baseline alignment**: Reproduce existing PyTorch vs H2O results (10k and full dataset) to confirm pipeline correctness.
2. **Feature stability**: Run MI/L1/GMB importances across forward-chaining folds; identify consensus subsets; evaluate size vs AUC/calibration trade-offs.
3. **Provider sensitivity**: Compare agnostic vs aware configurations; quantify calibration drift and expected utility under the same threshold policy.
4. **Loss functions & calibration**: Train BCE vs focal variants; apply post-hoc calibration; record Brier/ECE improvements.
5. **Threshold strategies**: Evaluate fixed vs Youden vs F1 vs utility thresholds on validation; measure transfer to test under base-rate shift scenarios.
6. **Neural vs boosting**: Tune MLP architectures (residual, embeddings) and compare to XGBoost/LightGBM/CatBoost; focus on contexts where MLPs close the performance gap.
7. **Robustness checks**: Stress tests, drift analysis, significance tests (bootstrap, DeLong) to ensure results are defensible.

Each experiment yields artifacts in `local_runs/` (metrics, curves, config snapshots) and summarized tables/figures for the thesis document.

## Engineering Tasks
- Extend configs for provider-aware runs and hybrid feature subsets (L1 + tree importances).
- Add a CLI/Make target for forward-chaining selection and calibration sweeps.
- Automate generation of metrics tables and plots for the thesis (ROC/PR curves, calibration curves, threshold trade-off diagrams).
- Ensure reproducible environments (lock requirements, log git SHA/dirty state).
- Track implementation-specific alignment work in `PROJECT_ALIGNMENT.md`.

## Relevance & Fit to the Project
- Aligns with the existing codebase (PyTorch pipeline, feature selection, H2O integration) and exploration documentation.
- Builds on current analyses (MI/L1 selection, H2O feature importance, SHAP), extending them into structured experiments.
- Supports thesis deliverables: clear research questions, methodology, evidence, and engineering foundations for reproducibility.
- Offers room for advanced extensions (survival modeling, reject inference, fairness, conformal calibration) if scope and time permit.

## Next Steps
1. Finalize the experiment calendar and resource plan (compute budget, expected runtime per config).
2. Prioritize feature stability and calibration experiments to produce early thesis figures.
3. Decide on optional extensions (e.g., reject inference, survival) based on available time after core deliverables.
4. Keep the proposal synced with code and documentation changes; ensure all final figures/tables are reproducible via Make targets.
