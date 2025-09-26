---
title: "Thesis Iteration 3 — Suggested Enhancements"
author: "Planning Notes"
date: "2025-09-26"
toc: true
number-sections: true
---

# Overview

This document collects concrete, implementation‑ready enhancements to extend Iteration 2. It retains the same evaluation invariants (time‑based split by `issue_d`, validation‑chosen fixed threshold, pos_label=0) and focuses on: temporal CV, calibration, neural upgrades in PyTorch, ensembling, and richer threshold analysis. It also outlines reproducible tables for split summaries and fairness monitoring.

# A. Neural Roadmap (PyTorch)

Architecture. Start with MLPs that combine residual blocks, GELU/SiLU activations, BatchNorm, and dropout. Introduce categorical embeddings for `grade/sub_grade`, `term`, `purpose`, and `addr_state`. Consider shallow attention over feature tokens once strong MLP baselines are in place.

Training protocol. Add temporal CV (expanding window); monitor AUCPR/ROC and thresholded metrics on validation; adopt cosine decay or OneCycle LR schedules with warmup; add early stopping with patience tuned via CV.

Calibration and thresholds. Fit Platt/Isotonic calibrators on the validation slice; choose thresholds via Youden J or utility‑optimized criteria; always freeze the threshold before scoring test.

Interpretability. Log permutation/SHAP varimp; corroborate with partial dependence/ICE on top drivers (`int_rate`, `dti`, `term`, grade) to validate monotonic trends.

# B. Practical Blueprint (Experiments)

1) Baseline MLP: 2–4 hidden layers (e.g., 256–128–64–32), BatchNorm after each hidden, dropout 0.2–0.4, AdamW optimizer, cosine‑annealed LR with warmup.
2) Replace one‑hot categoricals with embeddings: `grade/sub_grade` (dim 4–8), `term` (dim 2), `purpose` (dim 8), `addr_state` (dim 8). Concatenate embeddings with normalized numerics.
3) Add residual connections (pre‑activation) to stabilize deeper stacks; use GELU activations.
4) Monitor validation AUCPR; early stop with patience 10–20 epochs; always select test threshold from validation.
5) Evaluate BCE vs focal loss (γ≈2, α tuned); prefer BCE for calibration; otherwise calibrate post‑hoc.
6) Quantify drift with PSI; adopt a retrain cadence and recheck calibration/thresholds per vintage.

# C. Ensembling Strategies

Stack/blend top GBM/XGBoost models and calibrated NN models. Keep the time‑aware protocol and fixed thresholds. Report AUCPR and thresholded metrics; assess drift stability.

# D. Threshold Analysis Enhancements

Add precision at fixed recall targets (e.g., 0.5, 0.7), top‑k precision, and a simple expected‑profit curve on validation with fixed costs/benefits. Transfer the chosen threshold unchanged to test. These views align model selection with operational objectives.

# E. Temporal CV (Expanding Window)

Define K folds with expanding windows (e.g., [2007–2013]→2014, [2007–2014]→2015, …). Aggregate AUCPR/ROC and thresholded metrics across folds; report mean±SD in `reports/cv_metrics.json`. Enable `train_full_after: true` to refit on the full training span after CV.

# F. Split Summary and Fairness Monitoring

Split summary. Emit a table with train/validation/test date ranges, row counts, and positive rates. Source these directly from the pre‑split cohorts to avoid leakage.

Fairness snapshot. Report thresholded precision/recall/FPR by selected groups (e.g., `home_ownership`, `term`, top `addr_state`s) on the test split using the fixed validation‑chosen threshold. Use counts and rates; avoid small‑N groups.

Notes on implementation. Reuse the pipeline’s resolved splits and evaluation artifacts to generate both tables, ensuring identical filtering (final statuses only) and pos_label handling.

Definitions and computation notes
- Priors
  - What: Class base rates — proportion of the positive class (Charged Off; `pos_label=0`) in each split.
  - Why: Documents imbalance and verifies the split did not distort prevalence.
  - How: `prior = positives / total` per split; also report `n`, `positives`, `negatives`.
- Fairness table
  - What: Thresholded performance by group (e.g., `home_ownership`, `term`, selected `addr_state`s) on test at the fixed validation‑chosen threshold.
  - Why: Surfaces disparities across groups at the actual operating point.
  - How: For each group value with sufficient support: compute `n`, `positives`, `positive_rate`, `tp`, `fp`, `tn`, `fn`, `precision = tp/(tp+fp)`, `recall = tp/(tp+fn)`, `FPR = fp/(fp+tn)`.
  - Minimal columns: `group`, `value`, `n`, `positive_rate`, `precision`, `recall`, `FPR`.
