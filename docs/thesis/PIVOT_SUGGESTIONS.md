# Thesis Pivot Plan — Analytic Proposal and Roadmap

This document reframes the project as a rigorous, MSc‑level thesis platform. It consolidates findings from the current codebase and exploration docs, proposes a clear thesis scope and hypotheses, and defines the methodology, experiments, and engineering work needed for reproducible, defensible results.

## Executive Summary (Pivot)
- Core pivot: study portability and robustness of neural credit‑risk models under temporal and provider‑related shifts, with stability‑driven feature selection, calibration, and utility‑aligned thresholding.
- Two contrasting settings: provider‑agnostic (no pricing/scoring fields) vs provider‑aware (includes `int_rate`/`grade`/`sub_grade` when enabled).
- Key contributions:
  - Time‑aware stability selection yielding small, robust feature sets with near‑maximal AUC.
  - Calibration analysis across vintages; quantify “calibration drift” and mitigation via post‑hoc calibration.
  - Utility‑aligned threshold selection and partial ROC in business‑relevant FPR ranges.
  - Clear conditions when deep tabular models (MLP+embeddings) approach gradient boosting performance.

Note on dataset: this thesis uses the LendingClub consumer installment loans data (2007–2018 vintages). The “accepted” file contains funded applications with final statuses; a separate “rejected” file contains declined applications with limited covariates. Labels are derived from final outcomes (e.g., Charged Off vs Fully Paid) with care for right‑censoring in recent vintages. Features used for prediction are strictly limited to information available at origination; post‑event fields (payments, recoveries, last_* dates, hardship/settlement) are treated as leakage and excluded end‑to‑end.

## Research Questions and Hypotheses
- RQ1: How does excluding provider‑specific features affect out‑of‑time generalization and calibration?
  - H1: Provider‑aware models score higher in‑distribution but exhibit larger calibration drift; provider‑agnostic models generalize more consistently across time or simulated provider shifts.
- RQ2: Can a time‑stable subset (15–30 features) maintain ≥95% AUC vs all features and reduce drift sensitivity?
  - H2: Temporal stability selection (MI+L1 ensemble across forward‑chaining folds) yields compact subsets with minimal loss in AUC, improved calibration stability, and lower PSI sensitivity.
- RQ3: What are the calibration behaviors of BCE vs focal loss, and how effective are post‑hoc methods (Platt/Isotonic/Temperature Scaling)?
  - H3: Focal increases recall but harms calibration (higher ECE); post‑hoc calibration restores Brier/ECE to competitive levels.
- RQ4: Do utility‑based thresholds outperform generic criteria (Youden/F1) under base‑rate shift?
  - H4: Thresholds optimized for expected utility on validation maintain superior expected value on the unseen test, especially when prevalence shifts.
- RQ5: When do tabular neural networks match GBDTs on this task?
  - H5: MLPs with residual blocks and categorical embeddings close the gap to CatBoost/LightGBM at scale, particularly with provider‑aware features; embeddings are key in the agnostic setting.

## Methodology (Rigorous and Reproducible)
Invariants to maintain (already in code/config):
- Time‑based split by `issue_d` with an untouched out‑of‑time test; carve validation from the training period only.
- Oversampling (if used) on train subset only; class weighting/focal as alternatives.
- Threshold chosen on validation using the configured strategy and applied fixed to test.
- Respect `eval.pos_label` (default 0 = Charged Off) and seed Python/NumPy/Torch/DataLoader workers.

Dataset and leakage:
- Use the curated thesis dataset (`data/raw/full/thesis_data_full.csv`) with provenance compared against Kaggle accepted (`docs/data/comparison_accepted_vs_thesis.md`).
- Drop all post‑origination/leaky fields throughout (payments, recoveries, last_* dates, hardship/settlement).
- Address right‑censoring by restricting to cohorts with sufficient observation or document the censoring policy explicitly.

Splits and tuning:
- Evaluation: single time‑based holdout (e.g., train ≤ 2017‑06, test ≥ 2017‑07 or defined in config).
- Tuning and selection: forward‑chaining temporal CV (e.g., [2007–2014]->2015, [2007–2015]->2016, [2007–2016]->2017); fix hyperparameters from CV; train once on full train and evaluate once on the untouched test.

Baselines and comparators:
- Linear/logistic (L1/L2), RF, XGBoost/LightGBM/CatBoost; vs neural models (MLP, Residual MLP, Wide&Deep, MLP with categorical embeddings). Optional FT‑Transformer for breadth.

Calibration and thresholding:
- Post‑hoc calibration on validation: Platt, Isotonic, Temperature Scaling. Report Brier, ECE/MCE; apply calibrated transform to test.
- Thresholds: fixed (0.5), Youden J, F1, and utility‑based (cost matrix). Always select on validation.

Metrics and significance:
- Global: ROC‑AUC (with DeLong tests), PR‑AUC/AP (bootstrap CIs), Brier score, ECE/MCE.
- Operating point: precision/recall/specificity/F1 and expected utility at the validation‑selected threshold.
- Cohorts: metrics by quarter/year to show drift; PSI for features across early/late cohorts.

Explainability and stability:
- SHAP/permutation importances for top models; partial dependence on key features.
- Feature stability: Jaccard similarity of selected subsets across temporal folds; report a stability score.

Threats to validity (mitigations):
- Right‑censoring — restrict or model; document.
- Leakage — explicit list + tests; enforce in preprocessing.
- Tuning leakage — temporal CV; single untouched test.
- Multiple comparisons — pre‑register primary hypotheses; control when reporting many ablations.

## Experiment Design (Matrices and Ablations)
Axes:
- Feature sets: provider‑agnostic vs provider‑aware; full vs stability‑selected subset.
- Imbalance handling: none vs class weights vs ROS(train‑only) vs focal.
- Architectures: MLP → Residual MLP → Wide&Deep → MLP+Embeddings; optional FT‑Transformer.
- Calibration: none vs Platt vs Isotonic vs Temperature Scaling.
- Threshold strategy: fixed 0.5 vs Youden vs F1 vs utility‑based.
- Validation regime: holdout vs temporal CV for tuning; always final untouched test.

Outputs per run (beyond current):
- `metrics_ci.json` (bootstrap CIs for AUC/AP), `brier_score.json`, `calibration.png` (pre/post), `cohort_metrics.csv` (by quarter), DeLong p‑values when comparing models, `threshold_metrics.csv` (already produced), plus `experiments.csv` ledger row.

## Architecture Expansion (Neural Focus)
- MLP baseline improvements: residual blocks, GELU/SiLU activations, layer norm as alt to batch norm; one‑cycle LR, SWA, gradient clipping.
- Wide&Deep: linear “wide” crosses + deep path; helps sparse/categorical interactions.
- Categorical embeddings: convert selected categoricals to indices and learn embeddings; concatenate with normalized numeric features; compare to OHE.
- Advanced (optional): FT‑Transformer/TabTransformer; evaluate compute/perf trade‑offs vs MLP/GBDTs.
- Losses: BCE (+class weights), focal (already implemented), and exploratory AUC surrogate (pairwise ranking) with calibration check.
- Uncertainty: MC dropout or small deep ensembles; quantify ECE/Brier improvements.

## Evaluation & Statistical Testing
- AUC differences: DeLong test for correlated ROC curves; report p‑values.
- AP/Brier CIs: nonparametric bootstrap on test predictions; store in `metrics_ci.json`.
- Thresholded predictions: McNemar test for error disagreement; useful when comparing operating points.
- Partial ROC/PR: focus on business‑relevant FPR bands; include in cohort plots.

## Reproducibility & Documentation Enhancements
- Thesis context
  - README: add “Thesis” section (degree, university, year, topic, objectives, dataset provenance, how to cite).
  - `docs/thesis/OVERVIEW.md`: problem statement, hypotheses, datasets, evaluation protocol, risks.
  - Add `CITATION.cff` and short “Data Ethics” note.
- Experiment ledger (cross‑run index)
  - File `reports/experiments.csv` with fields: `run_id,timestamp,git_sha,config_path,backend,seed,n_train,n_test,class_prevalence,roc_auc,ap,threshold,pos_label,duration_sec,notes`.
  - CLI `src/cli/experiments_index.py` to build/append; cite ledger in thesis tables.
- Per‑run artifact upgrades
  - Already strong: `metrics.json`, `confusion.json`, curves, `features.json`, `requirements.freeze.txt`, `data_manifest.json`.
  - Add: `metrics_ci.json`, calibration artifacts, `cohort_metrics.csv`; expand run README with “Threats to validity” and related ADRs.
- ADRs (rationale tracking)
  - Keep accepted/proposed/rejected structure; add template; finalize ADRs for positive class convention and feature‑selection stability; reference ADR IDs in run READMEs.
- Sweeps and matrices
  - `experiments/*.yaml` define grids; CLI `src/cli/sweep.py` executes with deterministic seeds (`base_seed + idx`) and appends ledger rows.

## Roadmap (Incremental and Testable)
1) Documentation framing: README “Thesis” section; add `docs/thesis/OVERVIEW.md`; add `CITATION.cff` and Data Ethics note.
2) Reproducibility scaffolding: implement experiments ledger and per‑run CIs/calibration/cohort metrics; enrich run README.
3) Validation rigor: finalize accepted ADRs (threshold on validation — already implemented; positive class convention; oversampling isolation).
4) Temporal CV + stability selection: implement MI+L1 ensemble with forward‑chaining; emit stability metrics; produce small robust subsets.
5) Architecture expansions: residual MLP, Wide&Deep, embeddings; optional FT‑Transformer; add config toggles.
6) Sweeps CLI and experiments YAMLs: encode matrices for ablations; Make targets to run and summarize; integrate ledger.
7) Statistical testing utilities: bootstrap CIs, DeLong, McNemar; emit to artifacts and thesis tables.

## Acceptance Criteria (Thesis‑Ready)
- Every run is self‑contained and thesis‑consumable: metrics, CIs, calibration, cohort metrics, provenance, seeds, resolved config, figures.
- `reports/experiments.csv` provides a single, authoritative index for tables and comparisons.
- Primary hypotheses (H1–H5) are answered with significance tests, CIs, cohort analyses, and ablations; threats to validity documented.

## Ownership / Next Actions
- Approve the pivot scope (RQ/H hypotheses) and confirm provider‑agnostic vs provider‑aware configurations to include.
- Approve ledger schema and artifact additions (CIs/calibration/cohorts) under `local_runs/`.
- I can implement steps (1)–(4) immediately, then iterate through architecture and sweep expansions.

---

## LendingClub‑Specific Directions (High‑Value Extensions)

Because LendingClub (LC) is widely analyzed, the thesis benefits from LC‑specific angles that go beyond plain classification.

- Reject inference and selection bias
  - Use the public “rejected loans” file to quantify selection bias (accepted vs rejected cohorts).
  - Compare augmentation/correction strategies: hard cutoff augmentation, fuzzy augmentation, parceling, and EM/Heckman‑style two‑stage correction; measure PD calibration, AUC/AP, and drift.
  - Evaluate out‑of‑time generalization after reject inference; report calibration and utility changes.

- Survival/time‑to‑default modeling
  - Formulate risk as time‑to‑event with censoring (event: charge‑off; censor: fully paid/matured). Labels are derived; features remain origination‑time.
  - Compare Cox PH, parametric AFT, discrete‑time hazards, and deep survival baselines; convert survival to PD@12/24/36m and compare to binary PD.

- Positive‑Unlabeled (PU) learning for censoring
  - Treat charged‑off as P and the rest as U (unlabeled) to account for right‑censoring in recent vintages.
  - Compare nnPU/biased‑PU vs BCE/focal; focus on calibration and AP/AUC on mature cohorts.

- Macro‑aware stress testing
  - Join monthly macro series by `issue_d` (e.g., unemployment, credit spreads). Stress‑test PD under historical/synthetic scenarios.
  - Compare provider‑agnostic vs provider‑aware under macro shocks; quantify calibration drift and expected utility.

- Causal inference on pricing/grade
  - Treat `int_rate`/`grade` as provider treatments; estimate incremental effect on default via two‑stage residualization/partialling‑out.
  - Assess whether pricing/grades add unique signal vs proxy other origination features; discuss portability and fairness.

- Monotonicity and governance
  - Train monotone GBDTs (LightGBM/CatBoost constraints) and monotone‑regularized/deep lattice networks to enforce domain‑rational monotonicity (e.g., FICO↓ risk, DTI↑ risk).
  - Report governance metrics: KS, Gini, Lift, PSI/CSI; compare to unconstrained models.

- Fairness and calibration across segments
  - Audit fairness by proxies (e.g., `addr_state`, `home_ownership`, income bins, `purpose`).
  - Evaluate equalized TPR/FPR and “fairness of calibration” (ECE per subgroup). Compare global vs per‑segment thresholds and utility trade‑offs.

- Conformal risk control
  - Add conformal calibration to control false‑alarm rates per cohort/subgroup; report guarantees and degradation under drift; show benefits of recalibration.

### LC‑Focused Implementation Plan
- Label/survival toolkit
  - Add a labeling module to construct mature cohorts and survival targets (durations/censoring) from LC fields; used strictly for label construction.
- Reject inference CLI
  - Script to join accepted/rejected covariates and run augmentation/correction strategies; produce side‑by‑side PD calibration curves and stability tables.
- Macro join + stress
  - Ingest monthly macro series keyed by `issue_d`; scenario runner to evaluate PD/utility under shocks; config toggles to enable.
- Monotonic constraints and lattice option
  - Add LightGBM monotone constraints; optional monotone regularizer for MLP; compare governance metrics.
- Fairness/conformal utilities
  - Per‑segment metrics and conformal calibration helpers; artifacts for guarantees over time and subgroups.

### LC‑Specific Experiments (Outline)
- Reject inference vs baseline PD; measure calibration/AP/AUC and drift on future cohorts.
- Survival vs binary PD at 12/24/36m horizons; compare calibration and utility.
- Macro stress tests comparing agnostic vs aware; expected utility and calibration drift.
- Fairness: global vs per‑segment thresholds; cost of constraints on utility.
- Monotone vs unconstrained models: accuracy, calibration, governance metrics.
