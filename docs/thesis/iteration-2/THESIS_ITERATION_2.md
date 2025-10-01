---
title: "Credit Risk Modeling Platform for LendingClub Default Prediction: Neural Networks, Feature Regimes, and Time‑Aware Evaluation"
author: "Konstantinos Lambropoulos"
date: "2025-09-25"
bibliography:
  - docs/thesis/bibliography/lendingclub_subtopics_white.bib
  - docs/thesis/bibliography/credit_risk_neural_networks_research_papers.bib
  - docs/thesis/bibliography/strongly_confirmed_sources.bib
  - docs/thesis/bibliography/unconfirmed.bib
link-citations: true
link-bibliography: true
csl: docs/thesis/csl/apa.csl
toc: true
number-sections: true
header-includes:
  - \setlength{\emergencystretch}{2em}
  - \usepackage{array}
  - \usepackage{longtable}
  - \usepackage{booktabs}
  - \usepackage{caption}
  - \usepackage{float}
  - \usepackage[section]{placeins}
float-placement-figure: H
---

# Abstract

We study default prediction on LendingClub (2007-2018) under a time‑aware evaluation that enforces chronological splits, strict leakage controls, and fixed thresholds chosen on validation. We compare three dataset scales (10k, 100k, full) and four feature regimes (compact to provider‑aware with pricing/grades) across model families including neural networks. Enriching features with `int_rate`, `grade/sub_grade`, and `installment` consistently improves AUCPR: +0.04 at 10k (0.460 vs 0.421 baseline), +0.03 at 100k (0.452 vs 0.427), and +0.03 on full (0.393 vs 0.366). Tree ensembles lead overall on larger tabular datasets [@shwartz2022tabular; @grinsztajn2022why]; we analyze why and outline a neural blueprint (categorical embeddings, monotonic cues, regularization, calibration) to narrow the gap. Research questions focus on (i) the impact of provider‑aware features, (ii) family‑level performance patterns, and (iii) size effects under temporal drift. We provide reproducible artifacts (leaderboards, PR/ROC, varimp) and a roadmap for neural‑first improvements and drift robustness.

# Introduction

Peer-to-peer lending platforms such as LendingClub have catalyzed a rich literature on credit risk modeling, feature selection, and evaluation under class imbalance, with influential baselines built on this specific dataset [@Emekter2015; @SerranoCinca2015; @Jagtiani2019FM; @Croux2020JEBO]. We address a concrete practical question: Which feature subsets and model families-including neural networks-yield the best performance in a realistic, time-aware evaluation? We adopt a split policy that mirrors deployment: train on older loans, test on newer ones, and select the decision threshold on a held-out validation slice carved from the training period.

1.1 Credit Risk and Why It Matters

Credit risk is central to financial stability, consumer welfare, and the efficient allocation of capital. For lenders and platforms, accurate assessment of default probability underpins pricing (interest rates), provisioning, and regulatory capital; for borrowers, it impacts access and affordability. In consumer credit, small improvements in discrimination and calibration translate into large changes in expected loss, approval decisions, and portfolio utility. P2P platforms such as LendingClub intensified research interest by publishing rich origination‑time datasets with final outcomes, enabling reproducible empirical work on determinants of default, model performance, and profit‑aligned scoring [@Emekter2015; @SerranoCinca2015; @SerranoCinca2016; @Jagtiani2019FM; @Croux2020JEBO].

Three practical constraints shape this thesis: (i) class imbalance and asymmetric costs make precision-recall a better primary objective than accuracy; (ii) concept drift across vintages (macroeconomic cycles, policy changes, borrower mix) mandates time‑based evaluation and validation‑chosen thresholds; and (iii) leakage control is non‑negotiable-post‑event fields (payments, recoveries, last_* dates, hardship/settlement) must be excluded so the model reflects information available at origination only. Against this backdrop, we compare feature regimes (portable vs provider‑aware), evaluate neural networks alongside strong tree ensembles, and emphasize calibrated, thresholded decisions consistent with operational use.

1.2 Why Neural Networks for Credit Risk

Neural networks (NNs) are universal function approximators trained end-to-end, with the flexibility to incorporate learned embeddings for categorical features (e.g., grades), non-linear transformations for numeric features, and additional modalities (e.g., free-text loan descriptions). For tabular credit data, NNs must be configured thoughtfully: categorical embeddings (instead of brittle one-hot dummies), monotonic cues on known risk drivers (e.g., higher interest rate correlates with higher risk), strong regularization (BatchNorm, dropout, weight decay), and calibrated outputs. With these ingredients, NNs can be competitive with, and sometimes surpass, boosted trees-especially as dataset size and feature richness grow [@li2022evaluation; @wang2024hybrid].

Our motivation is that credit platforms continuously evolve underwriting criteria, borrower mix, and pricing, creating concept drift across vintages. Benchmarks that rely on random splits overstate performance by mixing future patterns into training. We therefore enforce chronological splits, careful leakage controls, and explicit thresholding on validation held out from the training period, allowing us to attribute improvements to modeling and features rather than evaluation artifacts and to assess stability over time.

## Contributions

- Reproducible, time‑aware evaluation framework with strict leakage policy, chronological splits, and fixed validation‑chosen thresholds; artifacts and Makefile‑driven runs are included in this repo.
- Systematic comparison of four feature regimes and three dataset scales; actionable evidence that provider‑aware features improve AUCPR at scale.
- Multi‑family baselines (GBM/XGB/DRF/GLM/NN) trained under a unified backend with aligned preprocessing; leaderboards, PR/ROC, and variable‑importance heatmaps.
- Drift analysis (PSI) and dataset‑size effects; practical mitigations for deployment (temporal CV, recalibration, recency weighting).
- Neural blueprint for tabular credit risk (embeddings, monotonic cues, regularization, calibration) tailored to LendingClub.

## Research Questions and Hypotheses {#sec:rq}

- RQ1: Under a time‑aware protocol, how do provider‑aware features (`int_rate`, `grade/sub_grade`, `installment`) affect AUCPR across dataset scales?
  - H1: Including pricing/grade improves AUCPR at 10k/100k/full relative to compact/broad‑without‑pricing regimes.
- RQ2: Which model families achieve the strongest discrimination under this protocol on tabular LC data?
  - H2: Gradient‑boosted trees outperform other families on larger tabular datasets; NNs can be competitive at smaller scales but lag without tailored encodings and calibration.
- RQ3: How does dataset size interact with temporal drift to influence out‑of‑time AUCPR and thresholded performance?
  - H3: AUCPR plateaus or declines at “full” vs 10k/100k due to drift across vintages; recency weighting and temporal CV mitigate this effect.
 

# Related Work

## Classical and Non‑Neural Credit Risk

Early empirical baselines on LendingClub analyze determinants of default and returns in P2P lending, establishing variables and evaluation setups that remain widely reused today [@Emekter2015; @SerranoCinca2015]. Subsequent work shows that tree ensembles (RF/GBM/XGB) and support vector machines provide strong tabular baselines for LendingClub default classification [@Malekipirbazari2015; @GuevaraDiaz2020; @NunezMora2023], while platform grades and alternative data emerge as influential predictors and policy signals [@Jagtiani2019FM; @Croux2020JEBO].

Researchers also propose profit‑aligned objectives and threshold selection strategies for P2P lending [@SerranoCinca2016], complementing default‑centric metrics and reinforcing our choice to tune thresholds on validation. Classic and modern studies frame default as a time‑to‑event problem, motivating explicit treatment of right‑censoring and temporal dynamics [@Banasik1999; @Bellotti2013; @SanchezBarrios2016]. Reject‑inference in survival contexts addresses sample‑selection bias when combining accepted and rejected cohorts [@Banasik2010], supporting our reliance on time‑based splits and caution around recent vintages.

Instance‑based decision support for P2P investors highlights feature design and evaluation schemes applicable to LendingClub [@Guo2016], and interpretability research shows how model‑agnostic tools such as LIME help stakeholders interrogate tabular credit models [@Ribeiro2016]. Regularized linear models (e.g., LASSO) and stability‑oriented selection remain standard for constructing compact, portable feature regimes in credit risk [@Tibshirani1996].

Together, the non‑neural literature establishes strong tree‑ensemble baselines on LendingClub, underscores the importance of pricing and grade variables, argues for profit‑aligned thresholding, and codifies the need for time‑aware evaluation to respect censoring and drift. Our setup adopts these invariants and uses them as a yardstick for neural models.

## Neural Networks and Deep Learning for Credit Risk

Neural credit risk spans classical MLPs for tabular data and modern deep architectures (CNNs, RNNs/LSTMs, attention/Transformers), increasingly fusing numeric features with text and alternative modalities. We organize this part by architecture family and modeling theme, roughly following historical progression.

### Deep MLPs for Tabular Credit Risk
Baseline tabular neural networks (feed‑forward MLPs) become competitive once preprocessing, categorical encodings, regularization, and calibration are handled carefully. Deployment‑focused case studies illustrate how such models integrate into risk and XVA stacks [@savine2022neural; @shen2021new], and social‑lending studies show that neural classifiers remain viable under imbalance when thresholds and calibration are tuned appropriately [@namvar2018credit; @jiang2022data; @emiroglu2018credit]. These findings motivate our emphasis on AUCPR and fixed validation‑chosen thresholds.

### Sequential CNN/LSTM and Temporal Deep Models
CNN‑LSTM hybrids and sequential deep learners have been applied to enterprise and bond defaults [@li2022evaluation; @wang2024hybrid] as well as financial monitoring tasks [@ala2020sequential]. Although LendingClub covariates are not per‑borrower time series, these works inform attention, gating, and regularization strategies that we can adapt to static tabular problems.

### Attention and Transformers for Credit Risk
Transformer‑based models increasingly power tabular and multi‑modal risk assessment [@huang2024enhancing; @wang2025research]. Attention mechanisms capture cross‑feature interactions without manual engineering, making them a promising direction for LendingClub‑like data when paired with regularization and monotonic constraints on known drivers such as `int_rate` and `dti`.

### Text Modeling (BERT/FinBERT) and Loan Descriptions
Textual fields (loan descriptions, job titles) provide complementary signals. Finance‑specific BERT variants and domain NLP studies suggest using pretrained encoders to enrich LendingClub models with text features [@hahn2024building].

### Generative and Data‑Augmentation Approaches
GANs and autoencoders can synthesize minority defaults to support neural training under imbalance [@van2023synthesizing; @lopez2020credit], and diffusion‑style augmentation is emerging for tabular data. Any augmentation must preserve temporal distributions and respect our leakage policy.

### Large Language Models (LLMs) and Generalist Scoring
Large language models are now explored for generalized credit scoring and GPT‑based classification [@boz2023generalist; @vasicek2024gpt; @feng2025explore], pointing toward end‑to‑end systems that leverage domain text and external knowledge. Even in these settings, evaluation must stay time‑aware and calibrated.

### Multi‑Modal and Multi‑View Deep Learning
Combining structured signals with text or alternative data often improves robustness and portability across providers [@al2023multi; @li2020multi]. For LendingClub‑like tasks this encourages adding text channels to neural baselines and calibrating the combined outputs.

### Surveys and Syntheses
Recent surveys emphasize careful data handling, leakage control, calibration under imbalance, temporal validation, and interpretable attributions [@ge2023credit; @fernandez2023complete]. These insights directly shape our neural blueprint: embedding categorical variables, encoding domain monotonicity for key features, comparing models via AUCPR, fixing thresholds on validation, and monitoring drift.

Taken together, the neural literature supports a neural‑first program that represents categoricals with embeddings, encodes monotone priors on features such as `int_rate` and `dti`, calibrates probabilities for threshold‑based decisions, validates temporally, and leverages pretrained text encoders when available.

# Dataset, Task, and Evaluation Protocol

## Problem Statement

Given origination‑time borrower and loan features X and a binary outcome Y indicating whether a loan ultimately charges off, learn a scoring function f: X -> [0, 1] that maximizes discrimination under class imbalance and supports calibrated, thresholded decisions out‑of‑time. We evaluate models with AUCPR (primary) and ROC AUC (supporting), choose a single operating threshold on a validation slice carved from the training period, and apply that fixed threshold to the test period.

## Dataset

We use the LendingClub consumer installment loans dataset spanning 2007-2018 vintages. Each record represents a funded loan at origination (accepted-loans cohort). Labels are derived from final outcomes: Fully Paid vs Charged Off (default). We adhere to origination-only features to avoid post-event leakage (e.g., payments, recoveries, last_* dates, hardship/settlement). Prior studies establish baseline determinants and modeling approaches on this dataset [@Emekter2015; @SerranoCinca2015; @Jagtiani2019FM; @Croux2020JEBO; @NunezMora2023].


Key columns used and their meanings are summarized below.
- `loan_amnt` (numeric): Amount borrowed.
- `term` (categorical: 36 or 60 months): Loan term.
- `int_rate` (numeric): Interest rate set at origination; a strong pricing proxy.
- `grade` / `sub_grade` (categorical): Platform credit grades; ordinal and highly informative.
- `installment` (numeric): Monthly payment amount; largely determined by `loan_amnt`, `term`, and `int_rate`.
- `annual_inc` (numeric): Stated annual income.
- `dti` (numeric): Debt-to-income ratio—a capacity/risk indicator.
- `fico_range_low` / `fico_range_high` (numeric): FICO range at origination; we also use `fico_avg` when engineered.
- `revol_bal` / `revol_util` (numeric): Revolving balance and utilization.
- `emp_length` (categorical): Employment length (binned); a stability proxy.
- `home_ownership`, `verification_status`, `addr_state`, `purpose` (categorical): Context and underwriting factors.
- `mort_acc`, `total_rev_hi_lim`, `num_rev_tl_bal_gt_0`, etc. (numeric): Depth and limits; credit capacity.

## Target

We frame the task as binary classification: predict whether a loan will charge off (default) versus fully pay. All metrics, curves, and thresholding treat Charged Off as the positive class.

## Chronological Split and Validation

We split by origination date (`issue_d`): earlier loans form training, later loans form test. Validation is carved from the training period only. This enforces causal ordering and avoids right-censoring leakage in recent vintages; standard random CV can mislead under temporal dependence [@bergmeir2018note]. We select a single decision threshold on validation using the Youden J statistic (maximizes TPR − FPR) [@youden1950index], then apply that fixed threshold to the test set for fair reporting.

## Metrics

- AUCPR (Average Precision): Summary of the precision-recall curve; sensitive to class imbalance and actionable for default detection [@saito2015precision; @davis2006relationship].
- ROC AUC: Threshold-independent ranking quality.
- Thresholded metrics at the selected operating point: precision, recall (TPR), FPR, confusion counts.

## Final-Status Filter and Censoring Cutoff

We restrict the cohort to loans with final outcomes at evaluation time to avoid right-censoring leakage. Specifically, we keep Fully Paid and Charged Off, and exclude operational/intermediate statuses (e.g., Current, Late, In Grace Period, Default/Issued).

::: {#tbl:final-status-filter}
| Status | Policy |
|---|---|
| Fully Paid | keep |
| Charged Off | keep |
| Current | exclude |
| In Grace Period / Late | exclude |
| Default / Issued / Other transitional | exclude |

: Final-status filter for the accepted‑loans cohort used in this study. We retain only loans with final outcomes (Fully Paid or Charged Off) at the evaluation cutoff to avoid right‑censoring leakage from in‑flight accounts. Removing intermediate statuses (e.g., Current, Late/Grace) ensures that labels reflect realized outcomes and that thresholded metrics on test align with deployment, where only origination‑time information is available.
:::

Our primary safeguard against censoring is the final-status filter, so we do not apply an additional calendar cutoff beyond the dataset’s coverage through 2018. This ensures reported performance reflects completed outcomes while retaining as much history as possible.

 

## Decision Thresholding and Business Metrics

In imbalanced credit settings, downstream utility depends on a fixed operating point (threshold). We choose the threshold on validation (carved from train) to avoid optimistic bias and apply it unchanged to test. We use Youden J (maximizes TPR − FPR) as a robust, distribution‑agnostic default [@youden1950index]; alternative strategies include F1 (balance precision/recall) or profit/expected‑value optimization when cost parameters are known [see @verbraken2014development].

::: {#tbl:full-confusion}
| Metric | Value |
|---|---:|
| Threshold (Youden J, validation) | 0.1765 |
| True Positives (tp) | 36,227 |
| True Negatives (tn) | 129,969 |
| False Positives (fp) | 68,284 |
| False Negatives (fn) | 19,876 |
| Precision | 0.347 |
| Recall (TPR) | 0.646 |
| False Positive Rate (FPR) | 0.344 |

: Thresholded confusion counts and derived rates on the full dataset at the single operating point selected on validation (Youden J) and transferred unchanged to test. This captures the business‑relevant trade‑off between catching Charged Off loans (recall/TPR) and avoiding false approvals (precision/FPR). Counts are impacted by class imbalance and by prevalence drift across vintages; use in tandem with AUCPR and ROC AUC.
:::

The table anchors the PR/ROC figures with the concrete operating point used for policy decisions. If utility or cost weights are available, the same summary supports expected‑value analysis to pick profit-optimal thresholds on validation and lock them for test.



## Leakage and Fairness Constraints (Definitions and Policy)

Leakage refers to any feature that contains information unavailable at origination or that lies causally downstream of the outcome. In the LendingClub data, that includes payments and recoveries, last payment dates, hardship or settlement flags, and collection‑stage balances; using them inflates apparent performance (see @fig:eda-corr-leaky). To prevent this, we drop post‑event fields end to end and restrict modeling to origination‑time variables (see @fig:eda-corr-orig and @fig:eda-psi-num). When ambiguity remains, we err on the safe side and omit the column.

We also guard against fairness issues and sensitive proxies. Demographic or geographic surrogates such as ZIP Code, while predictive, can generate disparate impact and hinder portability, so we omit them by default and focus on underwriting-relevant signals (capacity, credit history, pricing). Coarse geography like `addr_state` remains, but granular ZIP-like signals are removed.

High-cardinality and noisy fields pose additional risks. Free-text or ultra-granular categoricals such as `emp_title` explode the one-hot space, add noise, and increase variance; unless robust encodings and strong regularization are available, we prefer omitted or coarsened versions like the binned `emp_length`.

In practice we apply these policies by dropping payments, recoveries, last_* dates, hardship and settlement indicators, and collection fees, as well as granular location fields for fairness and portability. We omit or coarsen noisy categoricals—`emp_title` is removed, `emp_length` is retained in binned form, `purpose` is included with monitoring, `addr_state` remains coarse, and `grade/sub_grade` appears only in provider-aware regimes.

 

# Dataset Exploration (EDA)

We provide an early, self-contained view of the dataset to ground modeling decisions. Each figure is chosen to answer a specific question about class balance, leakage risks, signal strength, or temporal stability; each is referenced in the text where we use the insight.

::: {#tbl:eda-snapshot}
| Metric | Value |
|---|---|
| Total raw loans | 2,104,542 |
| Final statuses kept | 1,271,779 (Fully Paid 1,020,444; Charged Off 251,335) |
| Non‑final dropped | 832,763 (e.g., Current 799,583; Late/Grace 30,373) |
| Positive class (Charged Off) | 19.76% overall |

: EDA — Snapshot of the accepted‑loans cohort after the final‑status filter. Summarizes population size, outcome mix, and key filters applied prior to modeling. The positive class is Charged Off; its prevalence sets the PR baseline and guides interpretation of precision at a given recall. These cohort characteristics remain consistent across subsequent analyses (EDA, modeling, and thresholding).
:::

::: {#tbl:eda-features}
| Category | Count | Notes |
|---|---:|---|
| Numeric features | 51 | includes engineered ratios (e.g., `fico_avg`, `fico_spread`, `income_to_loan_ratio`) |
| Categorical features | 21 | grade/sub_grade, term, purpose, home_ownership, verification_status, addr_state, etc. |
| Parse‑as‑date | 2 | `issue_d` (split), `earliest_cr_line` (credit history) |

: EDA — Column types and scale after leakage‑aware selection. Numeric features include capacity and credit history measures; categorical features include grade/sub_grade, term, purpose, and state. Understanding type/scale mix informs encoding choices (one‑hot vs embeddings), regularization, and monotone cues; it also highlights columns to monitor for drift.
:::

Class balance over time (why shown). Default base rates shift materially across vintages; @fig:eda-class-balance makes explicit the trend we must respect with time‑based validation and fixed thresholds selected on validation.

![Class balance over time (positive = Charged Off). This figure shows the yearly prevalence of the positive class under the time‑based protocol (train on older vintages, validation carved from train, test on newer vintages). The baseline of the Precision–Recall curve equals this prevalence; rising default rates into 2016–2017 followed by a 2018 dip (right‑censoring/volume effects) explain why we emphasize AUCPR and fix thresholds from validation rather than tuning on test. Interpreting model precision at a given recall must account for these shifts; deployment should monitor prevalence and re‑assess thresholds when the base rate changes.](../../exploration/figures/class_balance_over_time.png){#fig:eda-class-balance}

Missingness and leakage (why shown). @fig:eda-missingness surfaces high‑missing, post‑event operational fields that must be excluded to avoid leakage.

![Top missingness by column (origination‑time perspective). The highest‑missing fields cluster around post‑event operations (e.g., hardship/settlement, recoveries, last payment dates). These variables are not available at origination and act as leakage if included, spuriously inflating metrics. Our policy drops such columns end‑to‑end so that models learn only from information available when the decision is made. Mild missingness in origination‑time fields is handled by the pipeline’s imputation steps.](../../exploration/figures/missingness_top.png){#fig:eda-missingness}

Distributions (why shown). Histograms contextualize ranges, outliers, and monotonic expectations—inputs to winsorization and monotone priors for NNs (see @fig:eda-hist-loan, @fig:eda-hist-int, @fig:eda-hist-fico, and @fig:eda-hist-dti).

![Loan amount distribution by class. This histogram contrasts loan sizes for Fully Paid vs Charged Off under the origination‑only feature set. Heavier right tails motivate winsorization and derived ratios (e.g., income‑to‑loan) to stabilize scale effects. While loan amount is informative, its contribution is moderated once pricing (`int_rate`) and term are included, since installment mechanically relates to these variables.](../../exploration/figures/hist_loan_amnt_orig.png){#fig:eda-hist-loan}

![Interest rate distribution by class. Higher interest rates associate with higher default rates, reflecting risk‑based pricing. This monotone relationship guides our use of monotonic cues for neural networks and explains why provider‑aware regimes (with `int_rate`) deliver sizeable AUCPR gains. Because pricing can drift with macro conditions and policy, we monitor this feature for distribution shifts over time.](../../exploration/figures/hist_int_rate_orig.png){#fig:eda-hist-int}

![FICO average distribution by class. Lower FICO aligns with higher default and remains a top origination‑time signal across dataset scales. The separation validates simple monotone expectations and supports mild winsorization to control outliers. FICO’s stability over vintages makes it a reliable baseline driver, complementing pricing and term.](../../exploration/figures/hist_fico_avg_orig.png){#fig:eda-hist-fico}

![DTI distribution by class. Debt‑to‑income (DTI) exhibits skew and heavier tails for Charged Off loans. This justifies winsorization and motivates soft monotonic regularization in NNs (higher DTI → higher risk, all else equal). Interactions with credit limits and utilization are captured naturally by tree ensembles and, with sufficient data, by NNs.](../../exploration/figures/hist_dti_orig.png){#fig:eda-hist-dti}

Categorical bar plots reveal ordinal monotonicity (grade/sub_grade), policy signals (term), and contextual drivers (purpose, home ownership) (see @fig:eda-cat-grade, @fig:eda-cat-subgrade, @fig:eda-cat-term, and @fig:eda-cat-purpose).

![Grade — counts and default rates. Default increases from A→G, with origination volume concentrated in B–D. Grade encapsulates provider policy and pricing; its ordinal structure is well suited to learned embeddings in NNs and to split ordering in trees. Because grade can drift with underwriting policy, production use should track its population stability.](../../exploration/figures/cat_grade_orig.png){#fig:eda-cat-grade}

![Sub‑grade — counts and default rates. Within‑grade monotonicity is smooth, making sub_grade a high‑signal categorical. For NNs, embeddings capture within‑grade proximity and interactions with other variables; for trees, ordered splits recover similar structure. As with grade, shifts in the sub_grade mix warrant drift monitoring.](../../exploration/figures/cat_sub_grade_orig.png){#fig:eda-cat-subgrade}

![Term — counts and default rates. 60‑month loans carry higher default risk than 36‑month loans, producing a crisp monotone split that tree models exploit efficiently. For NNs, one‑hot or small embeddings suffice; interactions with `int_rate` and loan size explain much of the term effect.](../../exploration/figures/cat_term_orig.png){#fig:eda-cat-term}

![Purpose — counts and default rates. Purpose categories capture heterogeneity in borrowing intent (e.g., debt consolidation vs small business). Signal is useful but exhibits modest drift over vintages; careful regularization and monitoring help maintain portability. Low‑volume categories should be grouped to avoid sparsity.](../../exploration/figures/cat_purpose_orig.png){#fig:eda-cat-purpose}

Two sets of correlation and PSI panels contrast origination‑only versus leaky features and quantify temporal drift (see @fig:eda-corr-orig, @fig:eda-corr-leaky, @fig:eda-psi-num, and @fig:eda-psi-cat).

![Top |corr| with target (origination‑only). Correlations computed on origination‑time numerics show strong anti‑correlation for FICO and positive associations for DTI/utilization. These relationships justify monotone cues and feature scaling choices; they also provide a leakage‑free sanity check for signal strength before modeling.](../../exploration/figures/top_corr_numeric_orig.png){#fig:eda-corr-orig}

![Top |corr| with target (all numerics). When post‑event fields are included, they dominate spuriously due to direct outcome information (e.g., recoveries), inflating apparent performance. This panel illustrates why we exclude such variables and restrict modeling to origination‑time data to prevent leakage.](../../exploration/figures/top_corr_numeric.png){#fig:eda-corr-leaky}

![PSI — numeric (origination‑only). Population Stability Index (PSI) by vintage highlights drift in depth/limit variables and moderate shifts in utilization. We treat PSI > 0.1 as moderate and > 0.25 as large; observed changes motivate time‑based validation and, in deployment, recalibration or retraining triggers tied to drift thresholds.](../../exploration/figures/psi_numeric_top_orig.png){#fig:eda-psi-num}

![PSI — categorical (origination‑only). Purpose exhibits modest drift over time, while grade/term mixes vary with macro conditions and platform policy. For production, we recommend PSI‑based monitors on pricing/grade and key operational categoricals, with a recalibration playbook when thresholds are breached.](../../exploration/figures/psi_categorical_top_orig.png){#fig:eda-psi-cat}

Taken together, Figures @fig:eda-class-balance–@fig:eda-psi-cat justify (i) time‑based splits and fixed thresholds, (ii) leakage exclusion policies, (iii) winsorization and monotone priors for NNs on `int_rate` and `dti`, (iv) embeddings for ordinal categoricals (grade/sub_grade), and (v) PSI-driven drift monitoring with recalibration or retraining.

# Feature Regimes and Dataset Scales

We evaluate four representative feature regimes:
1. Compact baseline (about 12 features): core demographic, capacity, and FICO-range signals.
2. Compact + pricing/grade (about 16 features): adds `int_rate`, `grade`, `sub_grade`, and `installment`.
3. Broad without pricing (about 39 features): adds depth/limits/utilization but excludes pricing/grade.
4. Broad + pricing/grade (about 43 features): combines broad signals and pricing/grade.

We run experiments at three dataset scales:
1. 10k: medium-sample; sufficient to benefit from richer features.
2. 100k: large-sample; strong signal and robust comparisons.
3. Full: largest cohort, closest to a production benchmark.

# Modeling Families and Why They Fit This Task

We compare common tabular modeling families and analyze their suitability to the LendingClub task.

Generalized linear models (GLMs) provide a transparent baseline with calibrated probabilities under certain assumptions, capturing additive effects but struggling with higher-order interactions unless engineered. Random forests reduce variance through ensembles of decorrelated trees and handle heterogeneous feature scales, though boosted trees often outperform them on tabular tasks. Gradient-boosted models such as GBM and XGBoost excel on structured data by capturing interactions with strong regularization and built-in handling of missingness, which explains their dominance in many credit risk studies. Fully connected neural networks approximate complex functions given sufficient data and regularization; they require careful preprocessing, categorical encodings, batch normalization, dropout, early stopping, and calibration to match boosted-tree performance, yet hybrid and carefully regularized variants have proven competitive on credit-like tasks [@li2022evaluation; @wang2024hybrid]. Neural networks matter here because they offer an end-to-end model that can learn embeddings for categorical grades [@guo2016entity], incorporate side-channel text (e.g., loan descriptions), and accommodate additional modalities in future iterations. With calibration and monotonic priors [@platt1999probabilistic; @zadrozny2001obtaining; @guo2017calibration; @chen2016xgboost; @ke2017lightgbm], they become more portable across providers.

## Feature Selection Procedure

Our feature-selection objective is to reduce variance and drift sensitivity while preserving predictive power under the same time‑based protocol used for training. The baseline workflow applies filter methods—mutual information (MI) and L1‑regularized logistic regression—as first-pass selectors. It follows a time-based split on `issue_d`, carves validation from the training period only, and mirrors training invariants such as imputation, winsorization, and encoding. MI captures non-linear dependency, whereas the L1 path favors sparse linear signal; comparing or aggregating them stabilizes rankings. We cap the shortlist by the target feature count (12/16/39/43 regimes) or an MI elbow, confirming AUCPR/ROC against the full set on validation. Outputs include the selected feature list, full ranking, and AUC/PR curves that drive the compact regimes in our experiments. Engineered features such as `fico_avg`, `fico_spread`, and `income_to_loan_ratio` can be toggled explicitly to quantify their lift, with selection runs mirroring training preprocessing so downstream metrics remain comparable.

## Feature Regimes: Provider‑Agnostic vs Provider‑Aware

The provider‑agnostic (portable) regime excludes pricing and scoring features such as `int_rate`, `grade`, `sub_grade`, and `installment`. This choice favors portability across lenders and reduces exposure to policy-driven drift; EDA confirms that, although these variables are predictive, they readily encode macro and policy effects. The provider‑aware regime keeps pricing and grade information, improving AUCPR/ROC at 10k/100k/full by leveraging monotone and ordinal signals. When deployment stays with the same provider, these fields capture underwriting decisions that correlate with risk, so we monitor drift with PSI and rely on calibration plus validation-chosen thresholds to maintain decision quality. Across regimes we balance accuracy against portability, monotonicity against policy sensitivity, and manage fairness considerations by avoiding granular geography such as ZIP codes. Our neural roadmap aims to close the agnostic gap through richer representations and monotone priors without sacrificing portability.

## Primer on Algorithm Families

Logistic regression maps a linear combination of inputs through a logistic link to produce probabilities; it is simple, interpretable, and fast to train, but it remains limited to additive effects unless interactions are engineered manually. Decision trees partition the feature space into regions with homogeneous labels, providing intuitive splits and basic nonlinearity at the cost of high variance. Random forests mitigate that variance through bagging and feature subsampling, handling mixed feature types but sometimes lagging tuned boosting methods in AUCPR. Gradient-boosted trees iteratively add weak learners to correct residuals, delivering strong performance on structured tabular data through shrinkage, subsampling, and depth constraints, albeit with heavier tuning requirements and no built-in monotonicity unless specified. Neural networks stack linear layers with nonlinear activations (e.g., ReLU or GELU) and often include batch normalization, dropout, and optimizers such as Adam/AdamW. They are flexible function approximators that can absorb learned embeddings and auxiliary modalities but remain sensitive to preprocessing, initialization, regularization, and probability calibration.

Binary cross-entropy is the default loss for probabilistic classification, while focal loss reweights hard examples to boost minority-class recall at the expense of calibration. Class weights or balanced batches help mitigate skew, so evaluation should prioritize precision–recall curves and AUCPR instead of accuracy. For threshold-dependent decisions such as approve/decline, calibrated probabilities are crucial; Platt scaling (logistic regression on logits), isotonic regression (non-parametric), and temperature scaling (for NNs) align predicted probabilities with empirical frequencies on validation.

# Experimental Setup

We enforce data handling and leakage control by excluding post-origination features (payments, recoveries, last_* dates, hardship or settlement indicators) from every run, consistent with best practices and prior empirical audits on LendingClub.

Preprocessing applies median imputation and standardization to numeric features, frequent-category imputation with one-hot encoding to categoricals, and winsorization to ratios prone to outliers (`dti`, `revol_util`, `income_to_loan_ratio`, etc.). When enabled, the pipeline adds engineered fields such as `fico_avg`, `fico_spread`, and `income_to_loan_ratio`.

Evaluation follows chronological train/test splits, selects thresholds on validation (Youden J) carved from the training period, and reports test metrics at that fixed operating point. We track AUCPR and ROC AUC alongside confusion counts, precision, recall, and FPR, following credit-risk guidance that stresses time-aware modeling and censoring controls [@Banasik1999; @Bellotti2013].

The automated modeling backend uses H2O AutoML to train GBM, XGBoost, DRF, GLM, and Deep Learning (MLP) models, yielding leaderboards and explainability artifacts (variable importance, partial dependence, SHAP-like insights). These outputs keep comparisons transparent and inform future NN engineering, while dedicated PyTorch runs are planned and described in Future Work.

All comparisons share the same dataset cohorts, origination-only features, chronological splits, validation-carved threshold selection, and consistent preprocessing. Each figure and table in this thesis references artifacts produced under these constraints, ensuring repeatability.

## H2O AutoML (How We Use It)

We leverage H2O as an industrial-strength modeling platform to establish strong baselines, standardized comparisons, and rich explainability. The toolkit ships first-class implementations across estimator families—GBM, XGBoost, DRF/XRT, GLM, and feed-forward neural networks—plus specialized algorithms such as survival/CoxPH, isolation forests, RuleFit, and target encoding. AutoML orchestrates these models under shared preprocessing and scoring policies, giving us an apples-to-apples environment for comparing neural networks with state-of-the-art tree baselines.

AutoML budgets can be expressed in time or model counts, leaderboards can be sorted by AUCPR (our primary metric under imbalance), and reproducibility is promoted through seeds, include/exclude lists, and cross-validation artifact retention. In this thesis we sort by AUCPR and scale the time budget with dataset size.

Explainability and comparison tools come built in: leaderboards, ROC/PR curves, per-family and per-model variable importance, permutation importance, partial dependence/ICE, and SHAP-like row explanations. We rely on these outputs for AUCPR/ROC comparisons, for interpreting drivers via per-family heatmaps, and for model-correlation or Pareto analyses that surface diversity and trade-offs.
<!-- Deployment artifacts detail removed to keep thesis self-contained -->

Using a single platform to produce multi-family baselines, curated leaderboards, and aligned explainability reduces variance in our comparisons and keeps the focus on the scientific questions—when neural networks compete, which features help them most, and how stable the conclusions remain across time splits. The figures and tables throughout the Results sections are generated from H2O outputs so that every claim is grounded in consistent, reproducible artifacts.

### Why We Chose H2O (Decision Rationale)

We adopted H2O as the comparative backend for four reasons that align with the thesis goals. First, it trains GBM, XGBoost, DRF, GLM, and DeepLearning models with consistent preprocessing, scoring, and logging, eliminating hidden confounders when comparing neural networks to ensembles; leaderboards are sorted by AUCPR to reflect our imbalance-aware objective. Second, the platform supplies rich, standardized explainability—per-family variable-importance heatmaps, partial dependence and ICE plots, and model-correlation or Pareto views—so we can interpret drivers and diagnose model diversity without bespoke code. Third, time-budgeted AutoML scales to larger datasets (100k and full) while keeping seeds and knobs (nthreads, include/exclude lists) reproducible. Fourth, these baselines complement the PyTorch neural roadmap that accompanies this thesis: H2O’s DeepLearning provides a strong, regularized MLP benchmark, while tree ensembles remain a robust yardstick, freeing the PyTorch track to focus on embeddings, monotone regularization, calibration, and temporal CV.

H2O’s DeepLearning is not a substitute for the latest tabular NN research (e.g., transformers with feature tokenization). We therefore treat it as a strong baseline and outline a PyTorch plan for neural-first advances, while acknowledging the Java dependency and mitigating it through pre-flight checks and containerized environments.

### DeepLearning (NN) Modeling Plan in H2O AutoML

In H2O AutoML 3.46.x, the DeepLearning (feed-forward NN) component consists of one small default model and three predefined hyperparameter grids. These steps live in the AutoML Java sources (`DeepLearningStepsProvider`) and appear in leaderboards with IDs such as `DeepLearning_def_1_AutoML_...` and `DeepLearning_grid_{1,2,3}_AutoML_...`, which we observe in our run artifacts (e.g., `docs/experiments/run/h2o_full_dataset/results/h2o_leaderboard.csv`).

- Default model (`def_1`). Hidden layers `[10, 10, 10]`; other parameters take H2O defaults (Rectifier activation, no dropout grid). Purpose: lightweight baseline NN.
- Grid steps (`grid_1`, `grid_2`, `grid_3`). Common base settings across all grids include RectifierWithDropout activation, adaptive rate (`_rho ∈ {0.9, 0.95, 0.99}`, `_epsilon ∈ {1e-6, 1e-7, 1e-8, 1e-9}`), input dropout ratios `{0.0, 0.05, 0.10, 0.15, 0.20}`, epochs set to 10000 (with early stopping on the AutoML metric), and hidden-layer dropout ratios shared per layer. Early stopping and validation frames follow the AutoML configuration (`stopping_metric: AUC`, `sort_metric: AUCPR` in our runs). Architectural grids vary by depth: `grid_1` searches one-layer widths `{20, 50, 100}` with dropout `{0.0 … 0.5}`; `grid_2` repeats the pattern for two layers; `grid_3` extends it to three layers.

These grids are defined in H2O AutoML’s Java code for release 3.46 (the version pinned in `requirements.txt`): `h2o-automl/src/main/java/ai/h2o/automl/modeling/DeepLearningStepsProvider.java`. The source sets the activation, adaptive-rate parameters, dropout grids, and hidden-layer sizes quoted above. Leaderboards showing entries like `DeepLearning_grid_2_AutoML_...` map directly to these steps; see the H2O AutoML and Deep Learning manuals for defaults and behavior [@h2o2018automl; @h2o2018deeplearning].

The search therefore explores shallow-to-moderate MLPs (1–3 layers) with modest widths (20/50/100) and systematic dropout or optimizer combinations. It omits embeddings, batch normalization, modern activations (e.g., GELU), and deeper stacks. We treat the results as a strong, regularized baseline for tabular data and build the PyTorch roadmap to extend beyond this regime with categorical embeddings, monotone regularization, calibration, and deeper architectures when justified by data scale.

## AutoML Settings (This Thesis)

::: {#tbl:automl-settings}
| Dataset | Max runtime | Sort metric | Seed | Families (eligible) | Threshold selection |
|---|---:|---|---:|---|---|
| 10k | ~300 s | AUCPR | 42 | GBM, XGB, DRF, GLM, DeepLearning | Youden J on validation |
| 100k | ~900 s | AUCPR | 42 | GBM, XGB, DRF, GLM, DeepLearning | Youden J on validation |
| full | ~5,400 s | AUCPR | 42 | GBM, XGB, DRF, GLM, DeepLearning | Youden J on validation |

: AutoML settings per dataset size, including training budgets, leaderboard sorting (PR vs ROC), and thresholding policy. Sorting by AUCPR emphasizes class‑imbalance‑aware ranking, while all models adopt a fixed operating threshold chosen on validation (Youden J) for test reporting. Settings are harmonized across sizes to support fair comparison.
:::

Budgets scale with dataset size (cf. suite run scripts), leaderboards are sorted by AUCPR to respect class imbalance, and thresholds are always chosen on validation before being fixed for test reporting.

# Results: Winners and Cross-Dataset Comparison

@tbl:winners summarizes the winning configuration (by AUCPR) per dataset size, along with ROC AUC. See per‑dataset figures in the relevant section.

::: {#tbl:winners}
| Dataset | Winner Family | Feature Regime | Avg Precision | ROC AUC |
|---|---|---|---:|---:|
| 10k  | GBM               | Broad+Pricing/Grade (43)     | 0.4601 | 0.7591 |
| 100k | XGBoost           | Broad+Pricing/Grade (43)     | 0.4524 | 0.7435 |
| full | GBM               | Broad+Pricing/Grade (43)     | 0.3934 | 0.7093 |

: Winners by dataset size (best AUCPR per size), including model family and feature regime. Use alongside PR/ROC figures to confirm envelope dominance and to understand how adding provider‑aware features (pricing/grade) shifts performance. Family clustering indicates whether gaps come from features, modeling, or both.
:::

See @tbl:winners for a compact overview; detailed curves and model explainability are analyzed next. We emphasize PR (precision–recall) as the primary metric due to class imbalance [@saito2015precision; @davis2006relationship]: it directly reflects precision at relevant recall levels for default detection. ROC AUC complements PR by showing overall ranking quality irrespective of threshold.

Across all three dataset sizes the broad + pricing/grade regime (43 features) wins by a clear AUCPR margin. Pricing (`int_rate`) and grade information consistently drive the lift, improving ranking quality and precision at the recall levels that matter for screening Charged Off loans.

## Ablation: Pricing/Grade Inclusion

Including provider‑aware features (`int_rate`, `grade/sub_grade`, `installment`) improves AUCPR consistently across scales:
- 10k: +0.0395 vs compact (0.4601 vs 0.4206).
- 100k: +0.0255 vs compact (0.4524 vs 0.4269).
- full: +0.0278 vs compact (0.3934 vs 0.3656).

These gains support H1 (the relevant section) and justify provider‑aware regimes when portability permits. Thresholded metrics at the fixed validation‑chosen threshold also improve precision at comparable recall (see per‑dataset sections).

# Per-Dataset Analyses with Inline Figures {#sec:per-dataset}

We now analyze each dataset size (10k, 100k, full), include curves and explainability figures, and interpret takeaways.

## 10k subset (medium-sample regime)

The broad + pricing/grade GBM leads the 10k benchmark with Average Precision 0.4601 and ROC AUC 0.7591. Figures \ref{fig:10k-pr} and \ref{fig:10k-roc} show the resulting PR and ROC envelopes: the validation-chosen operating point lies in a region where precision stays markedly higher than in the compact regimes for the same recall, while the ROC curve confirms stable ranking across thresholds.

![10k — Precision–Recall curve (winner). This panel shows the PR curve on the 10k subset for the winning GBM trained with the Broad+Pricing/Grade regime (43 features). The positive class is Charged Off; the horizontal baseline equals the test prevalence. The fixed operating threshold, selected on validation (Youden J), lies on the winner’s envelope where precision remains meaningfully higher at the same recall compared with compact regimes, implying fewer false approvals for a given catch rate.](reports/10k/figures/pr_curve.png){#fig:10k-pr}

![10k — ROC curve (winner). The ROC plot complements PR by assessing ranking quality independent of threshold. The GBM winner achieves high ROC AUC, indicating stable ordering of applicants across operating points. Combined with a validation‑chosen threshold, this supports transferring the operating point to the test period without overfitting to a particular prevalence.](reports/10k/figures/roc_curve.png){#fig:10k-roc}

![10k — Leaderboard (PR‑sorted). The PR‑sorted leaderboard ranks H2O models by AUCPR on the 10k subset. GBM leads with the Broad+Pricing/Grade regime, followed by other tree ensembles and then the neural baseline. Higher AUCPR reflects better precision across recalls and aligns with the operational objective of screening Charged Off.](reports/10k/figures/h2o_leaderboard_pr.png){#fig:10k-lbpr}

![10k — Variable‑importance heatmap (winners). Relative importances are normalized per model (GBM/XGBoost by split gain; DeepLearning by sensitivity). Pricing (`int_rate`), term, and grade/sub_grade dominate, with DTI and credit depth adding lift. This pattern motivates using provider‑aware features when portability permits and suggests embedding‑based encodings for NNs to better exploit ordinal structure.](reports/10k/figures/h2o_varimp_heatmap_winners.png){#fig:10k-varimp}

The PR leaderboard in Figure \ref{fig:10k-lbpr} makes the magnitude of improvement tangible: enriched features dominate the envelope rather than scoring a narrow win at one threshold. Variable-importance heatmaps (Figure \ref{fig:10k-varimp}) reinforce the story—`int_rate`, term, and grade/sub_grade account for most of the gain, with DTI and credit depth providing supporting signal. GBM/XGBoost importance reflects cumulative split gains, whereas the neural model’s sensitivity scores distribute mass across categorical partitions, hinting that embeddings or monotone regularization would help the NN exploit pricing/grade with the same sharpness as the tree ensembles.

## 100k subset (large-sample regime)

The 100k benchmark is led by XGBoost on the same broad + pricing/grade feature set, achieving Average Precision 0.4524 and ROC AUC 0.7435. Figures \ref{fig:100k-pr} and \ref{fig:100k-roc} show that richer samples preserve the PR lift while keeping ranking strength high, indicating that the gains are not confined to a narrow operating point.

![100k — Precision–Recall curve (winner). The XGBoost winner (Broad+Pricing/Grade, 43 features) achieves a wider PR envelope on the 100k subset, maintaining higher precision at relevant recalls. With more data, the model leverages interactions among pricing/grade, term, and capacity signals without overfitting, improving screening for Charged Off at fixed review capacity.](reports/100k/figures/pr_curve.png){#fig:100k-pr}

![100k — ROC curve (winner). Strong ROC AUC confirms robust ranking for the XGBoost winner. Paired with a validation‑selected threshold, this supports consistent Charged Off decisions on the test period and improves robustness to slight prevalence changes.](reports/100k/figures/roc_curve.png){#fig:100k-roc}

![100k — Leaderboard (ROC‑sorted). ROC‑sorted rankings on the 100k subset show XGBoost at the top with tree ensembles clustered closely, indicating similar ranking quality across families. This supports downstream thresholding choices derived on validation for Charged Off identification.](reports/100k/figures/h2o_leaderboard_roc.png){#fig:100k-lbroc}

![100k — Variable‑importance heatmap (winners). Importance concentrates further on pricing (`int_rate`), term, and grade/sub_grade at this scale, with DTI and credit limits contributing incremental lift. NN attributions appear more distributed across sub‑grades and states, consistent with learned embeddings capturing finer‑grained structure.](reports/100k/figures/h2o_varimp_heatmap_winners.png){#fig:100k-varimp}

The ROC leaderboard in Figure \ref{fig:100k-lbroc} highlights how ensembles cluster near the top, reinforcing that AUCPR gains coincide with solid ranking power. Variable-importance patterns become even more concentrated on pricing, term, and grade (Figure \ref{fig:100k-varimp}), with DTI and credit limits contributing incremental lift. Neural attributions place similar emphasis on grade/term and `int_rate`, but they spread weight across subgrades and states, reflecting broader embeddings; trees retain sharper splits and therefore sustain a small performance edge.

## Full dataset (production-like benchmark)

On the full cohort, the GBM with broad + pricing/grade features achieves Average Precision 0.3934 and ROC AUC 0.7093. The validation-selected threshold is 0.1765, yielding test confusion counts tp=36,227, tn=129,969, fp=68,284, fn=19,876 (precision 0.347, recall 0.646, FPR 0.344). PR and ROC curves (Figures \ref{fig:full-pr} and \ref{fig:full-roc}) jointly document performance: PR remains the primary business lens under imbalance, while ROC confirms that the threshold transfers without undue sensitivity to minor prevalence shifts.

![Full — Precision–Recall curve (winner). On the full cohort, the GBM winner (Broad+Pricing/Grade) forms the widest PR envelope. The fixed operating threshold (from validation) lands on a region of the curve that balances catch rate and false approvals in a way consistent with deployment. Because prevalence and mix drift over the long time span, PR is the primary lens for business‑aligned performance.](reports/full/figures/pr_curve.png){#fig:full-pr}

![Full — ROC curve (winner). ROC complements PR by confirming ranking stability at production scale across thresholds. High ROC AUC supports confidence that the validation‑chosen threshold can be transferred to the test period without excessive sensitivity to small shifts in score distributions.](reports/full/figures/roc_curve.png){#fig:full-roc}

![Full — Leaderboard (PR‑sorted). The PR‑sorted leaderboard for the full dataset shows GBM at the top with the Broad+Pricing/Grade regime, translating into fewer false approvals at comparable recall for Charged Off. Close clustering among tree ensembles indicates that the choice of boosting library matters less than feature regime and evaluation protocol.](reports/full/figures/h2o_leaderboard_pr.png){#fig:full-lbpr}

![Full — Leaderboard (ROC‑sorted). Tree ensembles dominate ROC on the full dataset, underscoring their strong ranking ability on tabular credit data. This supports reliable threshold selection on validation and stable performance out‑of‑time, even as prevalence varies.](reports/full/figures/h2o_leaderboard_roc.png){#fig:full-lbroc}

![Full — Variable‑importance heatmap (winners). At production scale, pricing (`int_rate`) remains the dominant driver with term and grade/sub_grade close behind; DTI and credit depth contribute secondary lift. NN attribution highlights finer granularity within sub‑grades and selected states/purposes, consistent with embedding‑based representations. Since pricing/grade reflect provider policy, portability requires monitoring drift and recalibrating as needed.](reports/full/figures/h2o_varimp_heatmap_winners.png){#fig:full-varimp}

Leaderboards in Figures \ref{fig:full-lbpr} and \ref{fig:full-lbroc} show that ensembles dominate both PR and ROC spaces, underscoring the stability of tree-based ranking on long-horizon tabular credit data. Feature importance (Figure \ref{fig:full-varimp}) again spotlights pricing, term, and grade/sub_grade, with DTI and credit depth as secondary contributors. Neural attributions elevate the same signals but spread weight across finer-grained subgrade and geography indicators, reinforcing the need for embeddings and monotone cues if the NN is to match the crisp splits that trees learn on monotone drivers. Diversity analyses (not shown) confirm that top models lie on a Pareto frontier for AUCPR and ROC, supporting ensemble or stacking strategies when incremental lift is desired.

# Why Ensembles Lead and How NNs Can Catch Up

9.1 Strengths of Gradient Boosting on Tabular Data

Boosted trees thrive on structured, heterogeneous tabular features: they naturally capture non-linearities and interactions without heavy feature engineering, handle missingness, and regularize effectively. Their built-in split search over ordinal encodings of categoricals (e.g., one-hot grade levels) yields powerful, piecewise-constant approximations that often set a strong bar.

9.2 Challenges and Opportunities for NNs on LendingClub

Neural networks must convert heterogeneously scaled, partially ordinal, and sometimes sparse inputs into stable representations. Learned embeddings replace wide one-hot encodings for high-cardinality fields (`sub_grade`, `addr_state`, `purpose`), preserving similarity structure and improving sample efficiency. Monotonic or domain-informed priors—such as constraining the effect of `int_rate` and `dti`—help stabilize training, reduce overfitting, and keep interpretations aligned with credit intuition. Robust regularization (BatchNorm, dropout schedules, weight decay) and careful optimization (learning-rate schedules with early stopping, mixup/cutmix, or sharpness-aware minimization) remain necessary complements. Because Charged Off is the positive class, calibrated losses or sampling strategies (focal loss, class weighting, balanced batches) are essential, with any oversampling confined to the training subset. Post-hoc calibration (Platt, Isotonic, temperature scaling) keeps probabilities trustworthy for precision–recall operating points, and forward-chaining temporal CV quantifies vintage-level variance so retraining and drift mitigation remain deployment-faithful.

# Dataset Size Effects and Temporal Drift

In several families the 10k subset even outperforms the 100k and full cohorts on AUCPR—counterintuitive until we account for temporal shift. As older vintages enter the training data, borrower mix, pricing policy, and macroeconomics drift away from the later test window. The fixed validation-chosen threshold then aligns less well with the test prevalence, so precision falls despite having more observations. Smaller samples drawn nearer to the test period maintain closer alignment and therefore produce higher AUCPR.

Concept drift (`int_rate`, grade policy, eligibility changes), prevalence shift, covariate shift in capacity/depth features, and label maturity all contribute to this pattern. To counteract them we combine expanding-window temporal CV, recency weighting, and drift-aware calibration. Additional safeguards include adjusting thresholds for changing base rates, preferring features that remain stable across folds (via PSI or selection frequency), and adding coarse time indicators or monotone constraints so neural networks respect known directional cues.

The recurring result—better AUCPR at 10k than at 100k or full—signals that drift can overwhelm simple sample-size gains. Temporal CV, recency weighting, and drift-aware calibration therefore become mandatory if we want to leverage the larger datasets without sacrificing out-of-time precision.

# Extended Analysis: Empirical Signals and Data Drift

Correlations at origination show FICO averages as strong anti-correlates (~−0.13) with default, while DTI and utilization are positively associated. Mutual information highlights `fico_spread`, `term`, `fico_avg`, `income_to_loan_ratio`, `loan_amnt`, and inquiry/depth features as high-signal drivers; these patterns appear clearly in Figure \ref{fig:eda-corr-orig}.

When we include post-event features such as `total_pymnt`, `recoveries`, or `last_pymnt_d`, correlations and MI spike spuriously (Figure \ref{fig:eda-corr-leaky}). The effect illustrates why these fields must remain excluded—they break causal ordering and inflate apparent performance.

Population-stability analysis (Figures \ref{fig:eda-psi-num} and \ref{fig:eda-psi-cat}) shows that credit depth and limit variables drift across vintages, while categorical mixes (e.g., `purpose`) shift more modestly. Pricing-related variables demand continuous monitoring and periodic recalibration.

Together these signals reinforce the need for time-based validation with a fixed, validation-chosen threshold, scheduled retraining, and PSI monitoring on top drivers so that probabilities and thresholds stay aligned as distributions move.

# Limitations and Threats to Validity

Right-censoring remains a threat to validity: recent vintages may be only partially observed, and although chronological splits reduce leakage they do not eliminate censoring artifacts. Survival or competing-risks modeling is reserved for future work.

Provider-aware features (pricing/grade) boost accuracy but can reduce portability across lenders or policy regimes. Provider-agnostic configurations sacrifice a small amount of AUCPR in exchange for better generalization out of domain.

Stated income and several categorical fields carry measurement noise; robust preprocessing and winsorization mitigate—but do not remove—the resulting bias and variance.

Search budgets and hyperparameters are intentionally bounded for reproducibility. Larger models, alternative regularization schedules, or tabular transformers may deliver further gains once the infrastructure scales.

The current iteration omits rejected applications and free-text fields. Both could influence selection bias estimates and incremental lift, so they appear in the future work roadmap.

Threshold selection uses Youden J on validation and transfers the resulting threshold unchanged to test. The approach balances sensitivity (Charged Off) and specificity without needing explicit cost weights, yet business settings with asymmetric utilities may prefer alternative criteria. Sensitivity checks against F1, precision at fixed recall, or simple expected-profit curves should accompany Youden J to ensure policy alignment.

Calibration curves and reliability metrics (e.g., Brier score, expected calibration error) are not yet reported. Without them, threshold transfer may deteriorate under drift. Subsequent iterations should fit calibration on the validation slice (Platt/Isotonic for trees, temperature scaling for NNs) and report post-calibration performance on test.

Finally, H2O’s DeepLearning importance is sensitivity-based and noisier than tree-based measures. Interpret NN importances qualitatively and corroborate them with partial dependence or ICE plots wherever possible.

# Conclusions

This iteration benchmarked default prediction on LendingClub across dataset scales (10k, 100k, full), feature regimes (compact vs broad, with/without pricing and grade), and model families (neural networks vs strong tree ensembles) under time-aware evaluation with validation-chosen thresholds.

Comparisons spanned three axes: dataset scale (10k, 100k, full cohorts), feature subset design (compact cores, broad depth/limit signals, and provider-aware variants with `int_rate`, `grade/sub_grade`, and `installment`), and model families (calibrated neural networks versus tree ensembles trained under identical splits and thresholding).

Key findings include the consistent lift from pricing/grade features [@serrano2015determinants; @emekter2015evaluating; @jagtiani2019roles], the stability of origination-time drivers such as FICO and DTI (reinforcing winsorization and monotone cues) [@chen2016xgboost; @ke2017lightgbm], and the impact of temporal drift observed via PSI, which necessitates time-based validation with thresholds fixed on validation [@siddiqi2006credit; @bergmeir2018note]. Tree ensembles remain the strongest performers on medium and large tabular datasets [@shwartz2022tabular; @grinsztajn2022why], while neural networks stay competitive on smaller samples and can close the gap by adding embeddings, strong regularization, monotone guidance, and calibration [@guo2016entity; @platt1999probabilistic; @zadrozny2001obtaining; @guo2017calibration]. Selecting a single operating point on validation and transferring it to test, coupled with calibration, delivers deployment-aligned metrics with stable thresholds.

In practice, production-like cohorts should start with the broad + pricing/grade regime and a boosted-tree baseline, continually monitor PSI, and recalibrate thresholds as drift appears. Teams pursuing neural-first approaches can narrow the remaining gap by pairing embeddings for high-signal categoricals with monotone priors on `int_rate`/DTI, robust regularization, temporal CV, and calibration—laying the groundwork for multimodal extensions such as text.

Overall, out-of-time precision hinges on disciplined temporal evaluation with fixed thresholds [@bergmeir2018note; @youden1950index], judicious inclusion of pricing/grade features when portability allows [@serrano2015determinants; @emekter2015evaluating; @jagtiani2019roles], and model choices that respect tabular structure and drift [@shwartz2022tabular; @grinsztajn2022why; @siddiqi2006credit]. These levers yield robust, interpretable gains and a clear roadmap for strengthening neural models.

# Future Work: High-Level Roadmap {#sec:future-work}

Near-term priorities include expanding-window temporal CV to quantify vintage-level stability, recency-aware validation for hyperparameter selection under drift, and post-hoc calibration (Platt/Isotonic for trees, temperature scaling for neural nets) accompanied by Brier/ECE reporting. These steps keep threshold transfer reliable as distributions shift. The PyTorch roadmap focuses on tabular-friendly neural upgrades—categorical embeddings for high-signal features, monotone regularization on `int_rate`/`dti`, strong regularization (BatchNorm, dropout, weight decay), and calibrated outputs—so that neural models remain interpretable while closing the gap with boosted trees.

Additional improvements involve calibrated ensembling of GBM/XGBoost and neural models, richer threshold analyses (precision at fixed recall, top-k precision, expected-profit curves) tied to operational objectives, and lightweight incorporation of text fields via pretrained encoders. Neural feature-selection techniques such as stochastic gates or hard-concrete layers will help learn compact, stable subsets under the same time-aware protocol.

Longer-term work explores policy-aligned, utility-optimized thresholds alongside AUCPR, drift-triggered recalibration/retraining pipelines driven by PSI, and uncertainty estimates that provide safeguards for regulatory or capital planning.



# Appendix A - Variable-Importance Tables (GBM winners)

These tables provide exact relative-importance percentages for the top features of the GBM models within each dataset (complementing the variable-importance figures shown in Sections 9.2-9.4). Percentages are normalized within each winner model.



::: {#tbl:a2-varimp-10k}
| Feature | Relative Importance (%) |
|---|---:|
| num__int_rate | 25.09 |
| cat__term_ 60 months | 7.64 |
| num__dti | 6.48 |
| num__loan_amnt | 6.35 |
| num__income_to_loan_ratio | 5.77 |
| cat__term_ 36 months | 4.58 |
| cat__grade_E | 3.76 |
| num__num_actv_rev_tl | 3.62 |
| num__credit_history_length | 3.51 |
| num__total_rev_hi_lim | 3.49 |
: Top variable importance for the GBM winner on the 10k subset. Importance reflects split‑gain contributions normalized within the model; it is not a correlation measure. Pricing (`int_rate`) and term dominate, with DTI and loan size contributing additional lift—consistent with economic intuition and with PR/ROC gains when provider‑aware features are present.
:::

::: {#tbl:a3-varimp-100k}
| Feature | Relative Importance (%) |
|---|---:|
| num__int_rate | 57.51 |
| cat__term_ 36 months | 10.71 |
| cat__term_ 60 months | 9.18 |
| num__dti | 2.95 |
| cat__grade_A | 2.24 |
| num__num_rev_tl_bal_gt_0 | 1.84 |
| num__tot_hi_cred_lim | 1.53 |
| cat__grade_B | 1.51 |
| num__loan_amnt | 1.47 |
| cat__grade_C | 1.34 |
: Top variable importance for the GBM/XGBoost winners on the 100k subset. With more data, importance concentrates further on pricing/grade and term, while capacity and depth metrics fill out secondary ranks. Interpret with drift in mind: provider policy and macro conditions can shift these distributions, warranting monitoring and recalibration.
:::

::: {#tbl:a4-varimp-full}
| Feature | Relative Importance (%) |
|---|---:|
| num__int_rate | 39.28 |
| cat__term_ 60 months | 17.89 |
| cat__grade_A | 6.39 |
| cat__term_ 36 months | 6.20 |
| cat__grade_B | 4.58 |
| num__dti | 3.91 |
| num__income_to_loan_ratio | 2.49 |
| num__fico_avg | 2.48 |
| num__mort_acc | 2.11 |
| num__annual_inc | 1.78 |
: Top variable importance for the GBM winner on the full dataset (production‑like benchmark). Pricing (`int_rate`) remains the primary driver, followed by term and grade bands; DTI and credit depth show stable but smaller contributions. This hierarchy aligns with underwriting practice and the PR/ROC edge observed for provider‑aware regimes.
:::

::: {#tbl:a5-common}
| Feature | Datasets (out of 4) |
|---|---:|
| cat__term_ 36 months | 4 |
| num__dti | 4 |
| cat__term_ 60 months | 4 |
| num__income_to_loan_ratio | 3 |
| num__int_rate | 3 |
| num__tot_hi_cred_lim | 2 |
| num__annual_inc | 2 |
| num__loan_amnt | 2 |
| cat__grade_B | 2 |
| cat__grade_A | 2 |
: Common drivers across dataset sizes (appear in at least two top‑10 lists). The recurrence of `term`, `dti`, and `int_rate` underscores their robustness as origination‑time signals. Shared drivers help define a portable core feature set, while provider‑specific fields (pricing/grade) require drift monitoring and recalibration for deployment across contexts.
:::

# Appendix B - Per‑Dataset Run Metrics (Exact Values)

These tables list all runs per dataset with exact metrics corresponding to the AUCPR/ROC plots shown above. “Features” is the count of input columns in the respective run; thresholds are the fixed values chosen on validation (Youden J) and applied to test. Unless otherwise stated, these are single‑run point estimates.



::: {#tbl:b2-10k}
| Run | Features | ROC AUC | Avg Precision | Threshold |
|---|---:|---:|---:|---:|
| run_20250925_023120 | 43 | 0.7591 | 0.4601 | 0.1487 |
| run_20250925_023823 | 16 | 0.7523 | 0.4264 | 0.2034 |
| run_20250925_021716 | 39 | 0.7467 | 0.4512 | 0.1315 |
| run_20250925_022418 | 12 | 0.7360 | 0.4206 | 0.3879 |
: 10k subset — exact test metrics for each evaluated configuration. Columns report feature count, ROC AUC, AUCPR, and the fixed threshold chosen on validation (Youden J) and applied to test. Use these alongside the 10k PR/ROC figures to interpret operating‑point trade‑offs; values are single‑run point estimates in Iteration 2 (no multi‑seed CIs).
:::

::: {#tbl:b3-100k}
| Run | Features | ROC AUC | Avg Precision | Threshold |
|---|---:|---:|---:|---:|
| run_20250925_032002 | 43 | 0.7435 | 0.4524 | 0.1783 |
| run_20250925_033737 | 16 | 0.7392 | 0.4452 | 0.1922 |
| run_20250925_030244 | 12 | 0.7252 | 0.4269 | 0.1652 |
| run_20250925_024526 | 39 | 0.7304 | 0.4419 | 0.1709 |
: 100k subset — exact test metrics for each evaluated configuration under the same time‑based protocol. AUCPR generally improves with richer features at this scale. Thresholds are fixed from validation to avoid overfitting to the test period; results are single‑run point estimates.
:::

::: {#tbl:b4-full}
| Run | Features | ROC AUC | Avg Precision | Threshold |
|---|---:|---:|---:|---:|
| run_20250925_070714 | 43 | 0.7093 | 0.3934 | 0.1765 |
| run_20250925_035452 | 39 | 0.7002 | 0.3839 | 0.1649 |
| run_20250925_053155 | 12 | 0.6815 | 0.3656 | 0.1725 |
| run_20250925_084408 | 16 | 0.6999 | 0.3825 | 0.1644 |
: Full dataset — exact test metrics for evaluated configurations on the production‑like cohort. AUCPR reflects performance under substantial temporal drift; ROC AUC confirms ranking stability. Thresholds are fixed from validation; consider sensitivity to small threshold shifts (±0.02) when interpreting precision/recall counts.
:::

# Appendix C - Neural Network (DeepLearning) Variable-Importance Tables

These tables show top features for H2O DeepLearning (NN) per dataset, normalized to percentages. They complement GBM tables in Appendix A and are referenced in Sections 9.2-9.4.



\begingroup\setlength{\tabcolsep}{4pt}\scriptsize\sloppy
::: {#tbl:c2-nn-varimp-10k}
| Feature | Relative Importance (%) |
|---|---:|
| num__fico_spread | 5.90 |
| cat__sub_grade_G1 | 5.80 |
| cat__sub_grade_G5 | 5.58 |
| cat__sub_grade_G4 | 5.43 |
| cat__purpose_renewable_energy | 5.29 |
| cat__addr_state_WY | 5.22 |
| cat__sub_grade_G3 | 5.09 |
| cat__addr_state_ND | 5.02 |
| cat__addr_state_VT | 5.00 |
| cat__addr_state_MT | 4.90 |
: Neural network variable importance (H2O DeepLearning) on the 10k subset using sensitivity‑based attribution. At this scale, attributions often spread across sub_grade levels, term, and engineered capacity features (e.g., `fico_spread`), reflecting embedding‑based representations and interactions captured by the network.
:::
\endgroup

\begingroup\setlength{\tabcolsep}{4pt}\scriptsize\sloppy
::: {#tbl:c3-nn-varimp-100k}
| Feature | Relative Importance (%) |
|---|---:|
| cat__grade_C | 6.75 |
| cat__home_ownership_MORTGAGE | 6.49 |
| cat__home_ownership_RENT | 6.29 |
| cat__sub_grade_A1 | 5.96 |
| num__fico_spread | 5.83 |
| cat__emp_length_10+ years | 5.71 |
| cat__purpose_credit_card | 5.33 |
| num__inq_last_6mths | 5.08 |
| cat__purpose_debt_consolidation | 4.92 |
| num__int_rate | 4.92 |
: Neural network variable importance on the 100k subset. With more data, the model elevates pricing/grade and term while retaining signal from capacity and geography. Attribution remains more diffuse than tree split‑gain, consistent with distributed embeddings; interpret with care across correlated categoricals.
:::
\endgroup

\begingroup\setlength{\tabcolsep}{4pt}\scriptsize\sloppy
::: {#tbl:c4-nn-varimp-full}
| Feature | Relative Importance (%) |
|---|---:|
| cat__sub_grade_A1 | 7.69 |
| num__int_rate | 6.65 |
| cat__sub_grade_A2 | 6.28 |
| cat__sub_grade_A3 | 6.01 |
| cat__sub_grade_A4 | 5.95 |
| cat__addr_state_CA | 5.70 |
| cat__purpose_debt_consolidation | 5.28 |
| cat__sub_grade_A5 | 5.01 |
| cat__sub_grade_E4 | 4.90 |
| cat__grade_C | 4.54 |
: Neural network variable importance on the full dataset. Focus remains on pricing (`int_rate`) and a hierarchy of sub‑grades alongside `addr_state` and `purpose`. The pattern complements GBM importance and highlights where embeddings capture within‑grade nuance that trees approximate via ordered splits.
:::
\endgroup

# Appendix D - Excluded / Included Columns Policy (Leakage, Fairness, Cardinality)

\begingroup\setlength{\tabcolsep}{4pt}\scriptsize\sloppy
::: {#tbl:d1-leakage}
| Column | Category |
|---|---|
| out_prncp | post‑event balance |
| out_prncp_inv | post‑event balance |
| total_pymnt | payments (leaky) |
| total_pymnt_inv | payments (leaky) |
| last_pymnt_d | last payment date (leaky) |
| last_pymnt_amnt | last payment amount (leaky) |
| next_pymnt_d | next payment date (leaky) |
| last_credit_pull_d | post‑event bureau pull |
| collection_recovery_fee | collections/recovery |
| recoveries | collections/recovery |
| hardship_flag / type / reason / status | hardship (post‑event) |
| hardship_amount / dates / length / dpd | hardship details |
| hardship_loan_status | hardship outcome |
| orig_projected_additional_accrued_interest | post‑event accrual |
| hardship_payoff_balance_amount | hardship payoff |
| hardship_last_payment_amount | hardship payment |
| debt_settlement_flag / date | settlement |
| settlement_status / date / amount / percentage / term | settlement details |
: Leakage columns identified and excluded end‑to‑end because they contain post‑event information (payments, recoveries, last_* dates, hardship/settlement). Including any of these would leak target information and inflate apparent performance; removing them enforces origination‑time modeling discipline.
:::
\endgroup

\begingroup\setlength{\tabcolsep}{4pt}\scriptsize\sloppy
::: {#tbl:d2-fairness-card}
| Column | Policy | Rationale |
|---|---|---|
| zip_code | exclude | fairness (granular geography), portability |
| emp_title (free text) | exclude | high cardinality/noise; use `emp_length` instead |
| addr_state | include | coarse geography acceptable; monitor drift |
| purpose | include | underwriting context; monitor drift |
| grade / sub_grade | aware only | policy/pricing signals; strong monotone/ordinal drivers |
| int_rate | aware only | pricing for risk; drift‑sensitive; monotone driver |
| installment | aware only | mostly deterministic from loan_amnt/term/int_rate |
: Fairness and cardinality policy examples. Sensitive proxies (e.g., granular geography) are omitted to reduce disparate impact and improve portability; high‑cardinality categoricals are consolidated or embedded to manage sparsity. The policy balances predictive performance with governance constraints.
:::
\endgroup

\begingroup\setlength{\tabcolsep}{4pt}\scriptsize\sloppy
::: {#tbl:d3-included}
| Column | Category |
|---|---|
| loan_amnt | loan design |
| term (36/60) | loan design (monotone risk) |
| annual_inc | capacity |
| dti | capacity (monotone risk) |
| fico_range_low/high; fico_avg | credit score |
| revol_bal; revol_util | utilization |
| open_acc; total_acc | credit depth |
| mort_acc; total_rev_hi_lim | capacity/depth |
| emp_length | stability proxy |
| home_ownership; verification_status; addr_state; purpose | context |
: Included origination‑time signals used for modeling: stable capacity, credit history, and selected policy variables available at origination. These provide the core signal outside provider‑specific fields; their distributions are monitored for drift (PSI) to maintain model reliability over time.
:::
\endgroup

When ambiguity remains, we prefer omission to avoid leakage and fairness concerns. In provider-aware regimes, pricing/grade can be included with monotone priors and calibration to manage drift; in portable regimes, excluding them improves generalization across lenders.

# Appendix E - H2O DeepLearning Hyperparameter Grids (Reference)

::: {#tbl:e1-dl-default}
| Component | Setting |
|---|---|
| Model | DeepLearning default (`def_1`) |
| Hidden layers | [10, 10, 10] |
| Activation | Rectifier |
| Early stopping | Enabled via AutoML settings |
: H2O DeepLearning default configuration used as a neural baseline. Specifies architecture, activation, regularization, and training defaults prior to any grid search. Serves as a reference for comparing tree ensembles vs neural models under identical evaluation settings.
:::

::: {#tbl:e2-dl-grids}
| Grid | Hidden choices | Hidden dropout ratios | Activation | Input dropout ratio | Adaptive rate (rho) | Epsilon | Epochs |
|---|---|---|---|---|---|---|---|
| grid_1 | [20], [50], [100] | [0.0] … [0.5] (single) | RectifierWithDropout | {0.0, 0.05, 0.10, 0.15, 0.20} | {0.9, 0.95, 0.99} | {1e−6, 1e−7, 1e−8, 1e−9} | 10000 (early‑stop bound) |
| grid_2 | [20,20], [50,50], [100,100] | [0.0,0.0] … [0.5,0.5] | RectifierWithDropout | {0.0, 0.05, 0.10, 0.15, 0.20} | {0.9, 0.95, 0.99} | {1e−6, 1e−7, 1e−8, 1e−9} | 10000 (early‑stop bound) |
| grid_3 | [20,20,20], [50,50,50], [100,100,100] | [0.0,0.0,0.0] … [0.5,0.5,0.5] | RectifierWithDropout | {0.0, 0.05, 0.10, 0.15, 0.20} | {0.9, 0.95, 0.99} | {1e−6, 1e−7, 1e−8, 1e−9} | 10000 (early‑stop bound) |
: H2O DeepLearning AutoML grid settings (3.46.x), outlining the search over widths, depths, activations, and regularization. Grid choices are constrained to remain comparable across dataset sizes; selected winners inform the NN entries in leaderboards and importance summaries.
:::

These grids and defaults originate in the H2O AutoML sources (`DeepLearningStepsProvider.java`, rel-3.46; our runs pin `h2o==3.46.0.7`). AutoML applies early stopping using the configured metric (leaderboards sorted by AUCPR, stopping on AUC), so `_epochs=10000` acts as an upper bound.

# Appendix F - Reproducibility and Environment

- Hardware/software: experiments run on a workstation with Python 3.10+, H2O `3.46.0.7`, and thread‑limited BLAS; figures rendered headlessly (`MPLBACKEND=Agg`).
- Seeds/determinism: seeds set across Python/NumPy/Torch/DataLoader workers; H2O seeded where applicable. All results in Appendix B are point estimates for single seeded runs unless noted.
- Makefile (selected targets):
  - `make explore CONFIG=...` - dataset EDA and leakage/missingness checks.
  - `make automl-h2o AUTOML_CONFIG=...` - H2O AutoML baselines (leaderboards, PR/ROC, varimp).
  - `make dryrun-h2o` / `make dryrun-h2o-cv` - smoke tests for single split / temporal CV.
  - `make run-catalog` / `make run-catalog-report` - index and summarize local runs.
- Config and invariants: chronological splits by `issue_d`; validation slice carved from training; post‑event leakage features excluded; threshold fixed from validation; AUCPR primary metric; ROC AUC supporting.
- Feature engineering used: `income_to_loan_ratio = annual_inc / loan_amnt` (with inf->NaN), `fico_avg = (fico_low + fico_high)/2`, `fico_spread = fico_high - fico_low`, `credit_history_length = issue_d - earliest_cr_line` in months.

Build notes: HTML/PDF are generated from this Markdown via Pandoc with `pandoc-crossref` and `--citeproc`; see `docs/thesis/iteration-2/README.md` for commands.

# References
