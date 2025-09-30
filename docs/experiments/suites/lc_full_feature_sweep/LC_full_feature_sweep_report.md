# LendingClub Full Feature Sweep — Comprehensive Findings

## 1. Experiment Context
- **Dataset & label policy**: All runs use the full LendingClub accepted-loans dataset with `Charged Off` mapped to label 0 (positive class) and `Fully Paid` to 1, as inherited from the shared defaults (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250927_143829/config_resolved.yaml:4`). Every pipeline disables leakage by dropping post-origination fields before modeling (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250927_143829/config_resolved.yaml:10`).
- **Split strategy**: Time-based hold-out on `issue_d` with the newest ~20% of vintages reserved for test evaluation (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250927_143829/config_resolved.yaml:106`). Validation is carved strictly from the training period to support threshold tuning.
- **Backend guardrails**: All runs rely on the H2O AutoML pipeline with class balancing, stochastic seeding, and a capped random forest of candidates. The first sweep (`run_20250927_143829`) enables the full algorithm suite minus stacked ensembles so we can inspect GBM/GLM/DRF/DeepLearning leaders (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250927_143829/config_resolved.yaml:158`). Later runs restrict `include_algos` to `DeepLearning` to focus on the neural family (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250928_224255/config_resolved.yaml:132`).

### Objective of This Review
The business goal is to maximize the capture of charged-off loans (true positives) while limiting missed defaults (false negatives). We compare exported models across five runs, inspect the algorithm families available in the first AutoML sweep, and interpret the temporal cross-validation results without conflating them with the main hold-out showdown.

## 2. Run-Level Metrics at the Deployed Threshold
The classification metrics and operating thresholds reported in `metrics.json` for each run are reproduced below. “CO” columns refer to the Charged Off class (label 0).

| Run | ROC AUC | PR AUC | Threshold | CO Precision | CO Recall | CO F1 | FP Precision | FP Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run_20250927_143829 | 0.699 | 0.383 | 0.159 | 0.867 | 0.625 | 0.726 | 0.333 | 0.662 |
| run_20250928_224255 | 0.694 | 0.368 | 0.170 | 0.863 | 0.643 | 0.737 | 0.336 | 0.638 |
| run_20250929_014759 | 0.680 | 0.353 | 0.171 | 0.857 | 0.636 | 0.730 | 0.326 | 0.624 |
| run_20250929_045221 | 0.697 | 0.374 | 0.179 | 0.871 | 0.599 | 0.710 | 0.326 | 0.686 |
| run_20250929_075827 | 0.698 | 0.371 | 0.181 | 0.871 | 0.607 | 0.716 | 0.330 | 0.683 |

**Interpretation**
- ROC AUC ranges between 0.68 and 0.70: all models rank borrowers similarly across the broad score spectrum despite configuration differences. This reinforces that incremental lift must be justified through finer measures (recall, precision, portfolio trade-offs).
- Average precision (area under the PR curve) mirrors ROC within ~0.03 spread. The first run’s GBM retains a slight lead in ranking quality, but the DeepLearning full deck (`run_20250928_224255`) sacrifices only ~0.015 PR AUC while delivering the highest Charged Off recall.
- Thresholds cluster tightly (0.158–0.181). The consistent range emerges from identical validation strategy and heavily imbalanced classes, signalling that the pipeline’s default thresholding logic behaves predictably across variants.

## 3. Confusion Matrices Reframed Around Charged Off Positives
The exported confusion matrices treat label 1 as positive. The table below reinterprets them with Charged Off (label 0) as the positive class to match our objective.

| Run | TP (CO) | FP (CO) | FN (CO) | TN (CO) | TP/FN Ratio |
| --- | --- | --- | --- | --- | --- |
| run_20250927_143829 | 123886 | 18975 | 74367 | 37128 | 1.666 |
| run_20250928_224255 | 127517 | 20290 | 70736 | 35813 | 1.803 |
| run_20250929_014759 | 126040 | 21113 | 72213 | 34990 | 1.745 |
| run_20250929_045221 | 118818 | 17621 | 79435 | 38482 | 1.496 |
| run_20250929_075827 | 120398 | 17777 | 77855 | 38326 | 1.546 |

**Key takeaways**
- `run_20250928_224255` is the only configuration that breaks the 1.80 TP/FN ratio, capturing roughly 3.6 k additional defaults relative to the GBM run for ~1.3 k extra false alarms. That directly advances the stated objective.
- Provider-aware L1 variations (`run_20250929_045221` and `run_20250929_075827`) improve precision marginally but at a steep recall cost (TP/FN ≈1.50). They are preferable only if the organization aggressively penalizes false positives (e.g., manual review capacity is scarce).

## 4. Run Narratives & Feature Importance
### 4.1 `run_20250927_143829` — Full AutoML Baseline
- **Config**: 38 engineered features spanning credit history, balances, purpose, and term length (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250927_143829/config_resolved.yaml:44`). AutoML explores GBM, GLM, DRF, and DeepLearning.
- **Outcome**: GBM export leads with ROC 0.699 / PR 0.383 (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250927_143829/metrics.json:29`).
- **Feature drivers**: Trees emphasise loan term, combined FICO averages, and DTI (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250927_143829/varimp_per_family/varimp_gbm.csv:2`). The DeepLearning variant gravitates toward purpose categories and regional effects (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250927_143829/varimp_per_family/varimp_deeplearning.csv:2`).
- **Interpretation**: Structured tree ensembles win the raw leaderboard because their monotonic dependence on term and FICO lines up with well-known underwriting heuristics. Yet recall is limited to 62.5 %, leaving headroom for neural variants that exploit high-cardinality categoricals more flexibly.

### 4.2 `run_20250928_224255` — DeepLearning with Full Deck
- **Config**: Same feature list and winsorisation as the baseline, but AutoML is restricted to DeepLearning grids (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250928_224255/config_resolved.yaml:43`).
- **Outcome**: ROC 0.694 / PR 0.368, but recall climbs to 64.3 % with precision holding at 86.3 % (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250928_224255/metrics.json:4`).
- **Feature drivers**: Education and debt-consolidation purposes, verification status buckets, and `fico_spread` rank highest (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250928_224255/varimp_per_family/varimp_deeplearning.csv:2`).
- **Why it works**: The network leverages one-hot encodings of purpose and verification categories that exhibit non-linear interactions with repayment behaviour, especially when the class-balancing oversample exposes rare segments. That flexibility surfaces defaults the GBM misses, explaining the higher TP/FN ratio despite slightly lower aggregate AUC.

### 4.3 `run_20250929_014759` — DeepLearning on L1 Core Features
- **Config**: 12-feature L1 subset removing pricing signals and higher-order credit metrics (`docs/experiments/suites/lc_full_feature_sweep/h2o/provider_agnostic_l1.yaml:5`).
- **Outcome**: ROC 0.680 / PR 0.353; recall slips to 63.6 %, precision to 85.7 % (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250929_014759/metrics.json:4`).
- **Feature drivers**: Term, purpose, credit-history length, and home-ownership categories dominate (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250929_014759/varimp_per_family/varimp_deeplearning.csv:2`).
- **Interpretation**: Removing secondary balance/limit features cuts recall by about 1 pp and PR AUC by 0.015. The model is leaner but sacrifices coverage, suggesting the truncated feature deck cannot fully discrimate borderline defaulters.

### 4.4 `run_20250929_045221` — Provider-Aware L1
- **Config**: L1 base augmented with lending-rate signals (`int_rate`, `grade`, `sub_grade`, `installment`) (`docs/experiments/suites/lc_full_feature_sweep/h2o/provider_aware_l1.yaml:5`).
- **Outcome**: ROC 0.697 / PR 0.374 and precision 87.1 %, but recall drops to 59.9 % (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250929_045221/metrics.json:4`).
- **Feature drivers**: Sub-grade bins and interest rate overwhelm the top-10 (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250929_045221/varimp_per_family/varimp_deeplearning.csv:2`).
- **Interpretation**: Pricing variables induce a sharp separation between high- and low-risk loans, which boosts precision. However, lending rates already embed lender expectations, so relying heavily on them can overfit to observed pricing policies and suppress recall in later vintages.

### 4.5 `run_20250929_075827` — Provider-Aware L1 with Temporal CV
- **Config**: Same feature set as §4.4 plus 5-fold expanding temporal CV with train_full_after (`docs/experiments/suites/lc_full_feature_sweep/h2o/provider_aware_l1_cv.yaml:5`).
- **Outcome**: Hold-out metrics mirror §4.4 (recall 60.7 %, precision 87.1 %), but CV folds show higher recall earlier in time (fold 1 recall 70.5 %, fold mean ROC 0.715) (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250929_075827/cv_metrics.json:4`).
- **Interpretation**: The fold-by-fold view reveals the model generalises well when scoring adjacent vintages, yet performance erodes in 2017–2018 due to portfolio drift. CV provides variance estimates (±0.02 ROC AUC, ±0.03 PR AUC), which set expectations for future improvements.

## 5. Algorithm-Family Snapshot from the Full AutoML Run
- **Leaderboard**: GBM, DeepLearning, GLM, and DRF all land within ~0.02 ROC AUC according to `h2o_leaderboard_test.csv` (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250927_143829/h2o_leaderboard_test.csv:1`). The GBM ensemble edges ahead with ROC 0.700 and PR 0.888, DeepLearning follows at 0.697 / 0.883, GLM 0.689 / 0.880, DRF 0.680 / 0.877.
- **Insights**:
  - Trees capture monotonic relationships with rates and FICO metrics, but plateau on high-cardinality categoricals.
  - GLM approximates a linear separating hyperplane; its high coefficients on total balance and DTI match intuition about leverage but lack interaction terms, explaining the lower recall.
  - DeepLearning matches GBM once the feature deck stays rich; its slight underperformance in the joint run likely stems from sharing the global AutoML time budget with tree grids.

## 6. Precision–Recall & Threshold Behaviour
- All runs’ `threshold_metrics.csv` show the optimal F1 point near the exported threshold (0.15–0.18), balancing recall ≈0.68 and precision ≈0.33 for the Fully Paid class. Lowering the threshold below 0.1 inflates recall but introduces extreme false-positive rates, while raising it above 0.5 obliterates recall in exchange for marginal precision gains.
- Sampling the precision–recall curves (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250928_224255/pr_points.csv`) reveals the DeepLearning full deck retains precision ≈0.34 at recall ≈0.60, similar to, but slightly better than, the GBM export. This indicates both models occupy nearly identical Pareto fronts, yet the neural model sits marginally higher in the high-recall region.

## 7. Temporal CV as a Diagnostic, Not a Leaderboard Entry
- The CV folds for `run_20250929_075827` confirm that the provider-aware network achieves 63–71 % recall on earlier time slices with thresholds around 0.18–0.20 (`docs/experiments/suites/lc_full_feature_sweep/h2o/run_20250929_075827/cv_metrics.json:34`). The drop to 60 % on the 2017–2018 hold-out underscores macro drift rather than modelling failure.
- Standard deviations (ROC ±0.020, PR ±0.030) supply noise baselines. Future feature or architecture tweaks must exceed these margins to be considered material.
- Fold thresholds provide a prior for production recalibration: even without a fresh hold-out set, we can start near 0.19 and fine-tune online.

## 8. Final Recommendations
1. **Adopt the DeepLearning full deck (`run_20250928_224255`) for Charged Off detection**. It delivers the best TP/FN ratio (1.803) while maintaining precision at 86.3 %. Confusion matrix comparison versus the GBM export demonstrates the practical gain: 3.6 k more defaults caught for 1.3 k additional false positives (Section 3).
2. **Retain GBM as a secondary benchmark**. Its ROC AUC lead (0.699 vs 0.694) signals slightly stronger global ranking. Monitoring both models in parallel can catch regressions and quantify the impact of future feature engineering.
3. **Leverage temporal CV for drift monitoring**. The expanding-window folds supply variance and threshold priors. Incorporate the CV run into scheduled health checks rather than the primary leaderboard.
4. **Feature insights**. Purpose and verification categories are pivotal for neural performance; removing them (L1) reduces recall, while over-relying on lender-set pricing (provider-aware L1) can harm coverage. Future work should enrich behavioural features rather than shrink the deck.

## 9. Appendix — Raw Feature Importance (Top 10)
Full listings for each run are available under `varimp_per_family/`. Highlights include:

### run_20250927_143829 — Full AutoML baseline (GBM leader)
This run enables the complete H2O AutoML algorithm suite on the 38-feature provider-agnostic deck; the exported model is a GBM ensemble that edged out DeepLearning/GLM/DRF candidates on the hold-out leaderboard.

| Feature | Relative Importance |
| --- | --- |
| cat__term_ 60 months | 77314.273438 |
| num__fico_avg | 24379.750000 |
| cat__term_ 36 months | 19051.445312 |
| num__dti | 16569.847656 |
| num__income_to_loan_ratio | 11184.063477 |
| num__fico_range_high | 9964.367188 |
| num__inq_last_6mths | 9044.012695 |
| num__total_bc_limit | 8817.603516 |
| num__fico_range_low | 8303.845703 |
| num__annual_inc | 7655.143555 |

### run_20250928_224255 — DeepLearning with full provider-agnostic deck
AutoML was restricted to DeepLearning, keeping the rich purpose and verification features to prioritise Charged-Off recall.

| Feature | Relative Importance |
| --- | --- |
| cat__purpose_educational | 1.000000 |
| cat__purpose_debt_consolidation | 0.865971 |
| cat__verification_status_Source Verified | 0.853987 |
| cat__verification_status_Verified | 0.821962 |
| cat__purpose_credit_card | 0.785283 |
| cat__verification_status_Not Verified | 0.776603 |
| cat__home_ownership_OTHER | 0.750018 |
| num__fico_spread | 0.662388 |
| num__bc_util | 0.540176 |
| cat__home_ownership_NONE | 0.536456 |

### run_20250929_014759 — DeepLearning on compact L1 subset
Uses the 12-feature provider-agnostic L1 core, exploring how a trimmed deck trades recall for simplicity.

| Feature | Relative Importance |
| --- | --- |
| cat__term_ 60 months | 1.000000 |
| cat__purpose_educational | 0.824068 |
| num__credit_history_length | 0.671981 |
| cat__term_ 36 months | 0.639428 |
| cat__home_ownership_OTHER | 0.613723 |
| cat__home_ownership_MORTGAGE | 0.579961 |
| cat__emp_length_10+ years | 0.577766 |
| cat__home_ownership_RENT | 0.567446 |
| cat__purpose_debt_consolidation | 0.558132 |
| num__dti | 0.546802 |

### run_20250929_045221 — Provider-aware L1 with pricing signals
Adds interest-rate, grade, and sub-grade features to the L1 deck, giving DeepLearning direct access to lender pricing cues.

| Feature | Relative Importance |
| --- | --- |
| cat__sub_grade_A1 | 1.000000 |
| cat__sub_grade_A3 | 0.895416 |
| cat__sub_grade_A2 | 0.839836 |
| num__int_rate | 0.824764 |
| cat__sub_grade_A4 | 0.810807 |
| cat__sub_grade_A5 | 0.756627 |
| cat__sub_grade_B1 | 0.736574 |
| num__dti | 0.688166 |
| cat__sub_grade_B2 | 0.663293 |
| cat__sub_grade_B3 | 0.563956 |

### run_20250929_075827 — Provider-aware L1 with 5-fold temporal CV
Matches the provider-aware L1 features but trains through expanding-window CV before refitting on the full history, surfacing temporal stability.

| Feature | Relative Importance |
| --- | --- |
| cat__grade_B | 1.000000 |
| cat__purpose_debt_consolidation | 0.930434 |
| cat__grade_C | 0.907054 |
| num__int_rate | 0.815500 |
| cat__purpose_credit_card | 0.780759 |
| cat__purpose_educational | 0.738531 |
| cat__addr_state_CA | 0.658995 |
| cat__emp_length_10+ years | 0.620544 |
| cat__sub_grade_A1 | 0.586472 |
| cat__home_ownership_OTHER | 0.558132 |
