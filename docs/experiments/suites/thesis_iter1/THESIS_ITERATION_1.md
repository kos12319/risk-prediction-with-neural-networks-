# Abstract

This thesis iteration investigates default risk prediction on the LendingClub consumer installment loans dataset (2007–2018), evaluating which feature subsets and modeling approaches—especially neural networks—yield the best precision–recall performance under a time‑based evaluation protocol. We compare compact baselines against enriched feature sets that include pricing/grade variables (`int_rate`, `grade`, `sub_grade`, `installment`) across dataset sizes (1k, 10k, 100k, full). Using an H2O AutoML backend, results show that enriched feature sets consistently improve AUCPR on medium and large samples (10k/100k/full), while smaller samples (1k) favor leaner feature sets to avoid overfitting. Tree ensembles (GBM/XGBoost) generally lead on larger datasets; deep neural networks are competitive on the smallest subset (1k) and remain a key part of the modeling toolkit. We conclude with guidance on feature selection by scale, thresholding protocol, and next steps to strengthen neural network performance and interpretability.

# 1. Introduction

Peer‑to‑peer lending platforms like LendingClub have catalyzed a rich literature on credit risk modeling, feature selection, and evaluation under class imbalance, with impactful baselines built on this specific dataset (Emekter et al., 2015; Serrano‑Cinca et al., 2015; Jagtiani & Lemieux, 2019; Croux et al., 2020). We study the problem of loan default prediction and, in particular: which feature subsets and model families—including neural networks—perform best under robust, time‑aware evaluation.

Objectives:
- Identify the best architecture per feature subset and dataset size under a time‑based split.
- Assess the contribution of pricing/grade variables to discriminative power.
- Evaluate the role and performance of neural networks (NNs) relative to tree ensembles.

Contributions:
- A configurable, reproducible experimental suite with Makefile‑driven runs and H2O AutoML backend to compare feature subsets at multiple scales.
- Per‑dataset reports with figures and metrics; a cross‑dataset summary synthesizing patterns and recommendations.
- Guidance on when enriched features (pricing/grade) help and when leaner sets are preferable, plus concrete next steps for improving NNs.

# 2. Background and Related Work

LendingClub default prediction has been widely studied as a benchmark for P2P credit modeling (Emekter et al., 2015; Serrano‑Cinca et al., 2015; Nunez‑Mora et al., 2023). Traditional baselines rely on logistic regression or tree ensembles (Malekipirbazari & Aksakalli, 2015), while more recent work explores richer features, profit‑aware objectives, and alternative data (Serrano‑Cinca & Gutiérrez‑Nieto, 2016; Jagtiani & Lemieux, 2019; Croux et al., 2020).

Neural networks for credit risk have shown promise when supplied with sufficient data and regularization, including deep MLPs and hybrid CNN/LSTM variants for sequence‑like signals (e.g., Li et al., 2022; Wang & Wang, 2024). However, on tabular data with strong monotonic signals, gradient‑boosted trees often excel out‑of‑the‑box, necessitating careful NN design, calibration, and feature engineering to compete.

Citations used in this iteration draw from the curated bibliographies under `docs/thesis/bibliography/`, including LendingClub‑focused and NN‑for‑credit‑risk sources.

# 3. Dataset and Labels

Dataset: LendingClub consumer installment loans, vintages 2007–2018, with funded loans and final statuses. Labels are derived from funding outcomes (e.g., Fully Paid vs Charged Off) and evaluated under a time‑based split by `issue_d` to mitigate leakage from right‑censoring of recent vintages.

- Positive class convention: `eval.pos_label=0` corresponds to Charged Off.
- Leakage policy: Post‑event fields (payments, recoveries, last_* dates, hardship/settlement) are dropped consistently.
- Splits: Older → train; newer → test. Validation is carved from the training period only.

References: Emekter et al. (2015); Serrano‑Cinca et al. (2015); Jagtiani & Lemieux (2019); Croux et al. (2020).

# 4. Methodology

4.1 Protocol and Invariants
- Time‑based split by `issue_d` (older→train, newer→test).
- Validation held out from the training period only; oversampling (if used) applies to train subset only.
- Threshold selection on validation using Youden’s J (unless otherwise configured); fixed threshold applied to test metrics.
- Determinism: seed Python/NumPy/Torch; consistent DataLoader seeding (for the PyTorch backend; here we use H2O backend for automl runs).

4.1.1 Data Sources and Cohorts
- Accepted loans (primary): `data/raw/full/thesis_data_full.csv` (see docs/data/DATA_SOURCES_REPORT.md).
- Rejected applications (limited covariates, no labels): `data/raw/full/kaggle_rejected_2007_to_2018Q4.csv` for reject‑inference research; not used for supervised training.
- Comparison to Kaggle accepted file shows thesis full is a curated subset with consistent schemas and missingness profiles; see docs/data/comparison_accepted_vs_thesis.md.

4.2 Experimental Suite and Reproducibility
- Makefile‑first: operations run via Make targets; configurations extend `configs/h2o/` and `configs/pytorch/` presets.
- Backend: H2O AutoML with leaderboard curves, per‑family variable importance, and optional partial dependence.
- Artifacts per run: metrics.json, ROC/PR curves, leaderboards, comparison figures, and varimp heatmaps.
- Suite structure: `docs/experiments/suites/thesis_iter1/` contains scripts, runs, and consolidated reports in `reports/`.

Architecture & ADR alignment:
- Time‑based split rationale (ADR 0001): docs/architecture/ADRs/accepted/0001-time-based-split.md.
- Threshold on validation (ADR 0004): docs/architecture/ADRs/accepted/0004-threshold-on-validation.md.
- Positive class convention (ADR 0011): `pos_label=0` = Charged Off; docs/architecture/journal/2025-09-24-standardize-positive-class-0.md.
- Makefile‑first policy (ADR 0013): docs/architecture/ADRs/accepted/0013-makefile-first-policy.md; journal: 2025‑09‑24 entry.
- Backend separation and pipeline design: docs/architecture/journal/2025-09-25-backend-pipeline-separation.md.

H2O capabilities leveraged:
- AutoML leaderboard and explainability (varimp, PDP/ICE, SHAP) per docs/h2o/LIBRARY_OFFERINGS.md, enabling consistent cross‑model comparison and deployment‑ready artifacts (MOJO/POJO).

4.3 Feature Subsets and Sizes
We evaluate feature sets at four dataset sizes: 1k, 10k, 100k, full. Feature subsets include:
- 12 features (compact baseline)
- 16 features (compact + pricing/grade)
- 39 features (broad credit/utilization/history; no pricing/grade)
- 43 features (broad + pricing/grade: `int_rate`, `grade`, `sub_grade`, `installment`)

4.4 Models
- H2O AutoML explores GBM, XGBoost, DRF, GLM, and Deep Learning (MLP) variants; we capture leaderboard performance and variable importance per family. Although this iteration emphasizes H2O to ensure a strong baseline, the suite is compatible with a PyTorch NN backend for future work.

# 5. Experiments

5.1 Datasets and Winners

Table 1 summarizes the winning run (by AUCPR) per dataset size, along with ROC AUC.

| Dataset | Winner Run | Features | Avg Precision | ROC AUC |
|---|---|---:|---:|---:|
| 1k   | run_20250925_020521 | 39 | 0.3148 | 0.7313 |
| 10k  | run_20250925_023120 | 43 | 0.4601 | 0.7591 |
| 100k | run_20250925_032002 | 43 | 0.4524 | 0.7435 |
| full | run_20250925_070714 | 43 | 0.3934 | 0.7093 |

Figure 1 compares AUCPR across feature sets within each dataset:
- 1k: `reports/1k/figures/aupr_by_feature_set.svg`
- 10k: `reports/10k/figures/aupr_by_feature_set.svg`
- 100k: `reports/100k/figures/aupr_by_feature_set.svg`
- full: `reports/full/figures/aupr_by_feature_set.svg`

Figure 2 compares AUCPR and ROC AUC across winners by dataset size:
- `docs/experiments/suites/thesis_iter1/reports/aupr_roc_winners_by_size.svg`

5.2 Full Dataset Focus

Winner: `run_20250925_070714` (43 features including pricing/grade). Test metrics at fixed threshold (Youden J on validation): AP 0.3934, ROC 0.7093, threshold 0.1765. Confusion: tp=36,227; tn=129,969; fp=68,284; fn=19,876 (Precision 0.347; Recall 0.646; FPR 0.344).

- PR curve: `reports/full/figures/pr_curve.png`
- ROC curve: `reports/full/figures/roc_curve.png`
- VarImp heatmap (winners): `reports/full/figures/h2o_varimp_heatmap_winners.png`
- Leaderboard PR/ROC: `reports/full/figures/h2o_leaderboard_pr.png`, `reports/full/figures/h2o_leaderboard_roc.png`

Top drivers (GBM family): `int_rate`, term (36/60), grade bands, `dti`, `income_to_loan_ratio`, `fico_avg`, `mort_acc`, `annual_inc`.

5.3 Model Families Observed at the Top

From the H2O leaderboards for the winner runs:
- 1k: DeepLearning (MLP) appears as the top model on the best 1k run.
- 10k: GBM leads.
- 100k: XGBoost leads.
- full: GBM leads.

Interpretation: Tree ensembles dominate at larger scales with these feature sets, while NNs can be competitive at very small scale (1k). This pattern aligns with known strengths of boosted trees on structured tabular data; NNs often require careful architecture/regularization, larger data, and feature learning to surpass ensembles.

# 6. Discussion: Feature Sets and Neural Networks

6.1 Feature Set Effects by Scale
- Enriched 43‑feature set (adds `int_rate`, `grade/sub_grade`, `installment`) consistently improves AUCPR on 10k/100k/full, vs. 12‑ or 39‑feature baselines.
- On 1k, the 39‑feature set outperforms the 43‑feature set; additional categorical expansion (grades, terms) increases variance and risks overfitting.

6.2 Neural Networks in This Iteration
- NNs (H2O DeepLearning) reached the top on the 1k benchmark, indicating strong potential on small sample regimes when regularization and compact features align.
- On larger datasets, GBM/XGBoost led. To strengthen NNs:
  - Architecture: tuned depth/width, batchnorm, dropout schedules; consider monotonic constraints or embedding strategies for high‑cardinality categoricals.
  - Training: robust early stopping, learning rate schedules, and calibration for threshold stability.
  - Features: engineered ratios (e.g., `income_to_loan_ratio`), standardized credit history length, careful handling of grade/sub_grade to avoid explosion of dummies (e.g., embeddings).
  - Validation: temporal CV (expanding window) to guard against vintage shifts.

6.4 Empirical Signals from Exploration
- Correlation (origination‑only numeric):
  - Image: `docs/exploration/figures/top_corr_numeric_orig.png`
  - Finding: FICO measures (avg/low/high) are strongest anti‑correlates (~-0.13); DTI and utilization are positively associated.
- Mutual Information (origination‑only):
  - Strong MI in `fico_spread`, `term`, `fico_avg`, `income_to_loan_ratio`, `loan_amnt`, and inquiry/depth features.
- Temporal Drift (PSI):
  - Images: `docs/exploration/figures/psi_numeric_top_orig.png`, `docs/exploration/figures/psi_categorical_top_orig.png`
  - Finding: Depth/limit features shift substantially across time; `purpose` shows modest drift; pricing variables require monitoring.
- Leakage audit (explicit): payments/recoveries, post‑event dates, hardship/settlement must be excluded (docs/exploration/EXPLORATION_REPORT.md).

6.3 Thresholding and Business Operating Points
- Threshold chosen on validation via Youden J provides a single fixed operating point for test. For deployment, evaluate PR operating points that align with cost/benefit trade‑offs and consider probability calibration.

# 7. Threats to Validity
- Right‑censoring in recent vintages: mitigated via time‑split by `issue_d`, but residual effects may remain.
- Class imbalance shifts over time: aggregated CV across time windows can better quantify variance.
- Leakage: strict dropping of post‑origination fields enforced; double‑check any engineered features for implicit leakage.

# 8. Conclusion

This iteration finds that pricing/grade features materially improve discrimination at moderate‑to‑large scales, while small samples benefit from compact, high‑signal features. Tree ensembles (GBM/XGBoost) lead on 10k/100k/full. Neural networks are competitive on the smallest benchmark and remain promising, especially with improved architectures and encodings for tabular categorical structure. Next iterations will target NN‑centric enhancements and cross‑time validation to close performance gaps.

# 9. Future Work
- Neural nets:
  - Add a dedicated PyTorch MLP pipeline with tuned hyperparameters, categorical embeddings, and monotonic regularization where justified.
  - Explore self‑supervised/contrastive pretraining for tabular signals.
- Evaluation:
  - Temporal CV with `train_full_after: true` for stability; calibration checks.
  - Profit‑aware metrics alongside AUCPR/ROC (Serrano‑Cinca & Gutiérrez‑Nieto, 2016).
- Features:
  - Standardize `credit_history_length`; expand ratio features and interaction terms subject to leakage policy.
- Robustness:
  - Stress tests on recent vintages; monitor drift in distributions and performance.

Feature selection track:
- Integrate the selector registry and artifacts (docs/feature_selection/FEATURE_SELECTION.md) so selection runs reuse preprocessing/evaluation from training.
- Quantify engineered feature lift via paired selection/training with engineered toggles; add stability analysis (bootstrap / time‑blocked resampling).

# References (selection)
- Emekter, R., Tu, Y., Jirasakuldech, B., & Lu, M. (2015). Evaluating Credit Risk and Loan Performance in Online Peer‑to‑Peer Lending. Applied Economics. (BibTeX: `Emekter2015`)
- Serrano‑Cinca, C., Gutiérrez‑Nieto, B., & López‑Palacios, L. (2015). Determinants of Default in P2P Lending. PLoS ONE. (`SerranoCinca2015`)
- Jagtiani, J., & Lemieux, C. (2019). The Roles of Alternative Data and Machine Learning in Fintech Lending. Financial Management. (`Jagtiani2019FM`)
- Croux, C., Jagtiani, J., Korivi, T., & Vulanovic, M. (2020). Important Factors Determining Fintech Loan Default. JEBO. (`Croux2020JEBO`)
- Nunez‑Mora, J. A., et al. (2023). Loan Default Prediction: A Complete Revision of LendingClub. REMEF. (`NunezMora2023`)
- Malekipirbazari, M., & Aksakalli, V. (2015). Risk Assessment in Social Lending via Random Forests. ESWA. (`Malekipirbazari2015`)
- Li, H., Sun, J., & Li, A. (2022). CNN‑LSTM‑ATT for Credit Risk. Electronics. (`li2022evaluation`)
- Wang, Y., & Wang, J. (2024). Hybrid CNN‑LSTM for Bond Default. J. Computational Science. (`wang2024hybrid`)

Notes: Full BibTeX entries are available under `docs/thesis/bibliography/`.
