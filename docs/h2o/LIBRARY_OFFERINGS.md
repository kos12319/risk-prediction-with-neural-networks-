# H2O Library Offerings — Notes for Project Extension

These notes capture key capabilities from the H2O Python package (referenced in `/private/tmp/h2o_pkg/h2o_wheel/...`) that are most relevant if we extend the thesis or project scope beyond the existing PyTorch pipeline.

## Platform & Data Management
- Connect to running H2O clusters and control sessions: helpers cover connection management, lazy frame import, SQL/Hive ingestion, Rapids scripting, and progress reporting (`init.py:26-38`).
- Handle end-to-end data flow with utilities such as `connect`, `import_file`, `parse_setup`, `rapids`, plus frame-creation and cleanup primitives, enabling reproducible experiments on remote or local clusters.

## Model Packaging & Deployment
- Export trained models via `download_pojo`, `download_model`, and `save_model`.
- Score models offline using `mojo_predict_csv` / `mojo_predict_pandas` for batch inference while keeping MOJO/POJO portability (`init.py:32-38`, `utils/shared_utils.py:59-152`).

## Estimator Catalog
- First-class algorithms across multiple families (`estimators/__init__.py:10-73`):
  - Tree ensembles (GBM, XGBoost, DRF/XRT).
  - Linear and generalized models (GLM, ANOVAGLM, HGLM).
  - Neural/deep learning (deep learning networks, autoencoders).
  - Survival (CoxPH), anomaly detection (IsolationForest, ExtendedIsolationForest), uplift/random forest, NLP (Word2Vec), unsupervised (KMeans, PCA, SVD), RuleFit, SVM, target encoding, and more.

## Automated Modeling
- H2OAutoML (`automl/_estimator.py:208-299`) orchestrates search across algorithms with controls for runtime/model budgets, early stopping metrics, reproducible seeds, include/exclude algorithm lists, exploitation vs. exploration budgets, custom modeling plans, target encoding, monotone constraints, CV artifact retention, and leaderboard sorting.

## Model Explainability & Comparison
- Built-in dashboards (`explanation/_explain.py:120-160`) surface:
  - Leaderboards and scoring summaries.
  - Confusion matrices, ROC/PR curves, and subgroup comparisons.
  - Variable importance, permutation varimp, PDP/ICE charts, SHAP row explanations.
  - Fairness metrics (AIR plots, ROC/PR by subgroup) and fairness-aware PDP/SHAP overlays.
- `make_leaderboard` and related helpers provide consistent cross-model scoring tables for manual or automated comparison.

## Pipeline Packaging
- `H2OMojoPipeline` bundles preprocessing with trained models for deployment, ensuring parity between AutoML pipelines and downstream scoring contexts (`pipeline/__init__.py:7-10`, `init.py:37-38`).

### Relevance to This Project
- The leaderboard tooling and explanation dashboards can complement our neural-network experiments by tracking multiple models (e.g., PyTorch baseline vs. H2O AutoML) and visualizing ROC/PR curves side-by-side.
- MOJO/POJO exports enable reproducible deployment artifacts that match any figures reported in the thesis.
- Automated modeling plus fairness reports can expand the scope when comparing provider-aware vs. provider-agnostic settings under the proposed research questions.
