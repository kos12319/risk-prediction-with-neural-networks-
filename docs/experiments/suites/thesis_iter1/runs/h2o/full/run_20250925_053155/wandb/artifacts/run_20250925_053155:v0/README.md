# Training Summary — run_20250925_053155

Config: `docs/experiments/suites/thesis_iter1/h2o/full/selected_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.172499

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 1468.2 KB |
| Start (UTC) | 2025-09-25T02:31:55+00:00 |
| End (UTC) | 2025-09-25T04:05:29+00:00 |
| Total time | 5613.83 s |
| Load | 11.16 s |
| Split | 0.27 s |
| Preprocess | 2.45 s |
| Train | 5582.39 s |
| Eval | 17.57 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.682
- Average Precision: 0.366
- Precision (at threshold): 0.326
- Recall (TPR): 0.625
- Specificity (TNR): 0.634
- Confusion: TP=35068, FP=72488, TN=125765, FN=21035
- n_train: 813938
- n_val: 203485
- n_test: 254356
- n_features: 96
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8567098092643052,
    "recall": 0.634366188657927,
    "f1-score": 0.7289604785351815,
    "support": 198253.0
  },
  "1": {
    "precision": 0.32604410725575517,
    "recall": 0.6250646133005365,
    "f1-score": 0.42854960619336546,
    "support": 56103.0
  },
  "accuracy": 0.6323145512588655,
  "macro avg": {
    "precision": 0.5913769582600301,
    "recall": 0.6297154009792318,
    "f1-score": 0.5787550423642736,
    "support": 254356.0
  },
  "weighted avg": {
    "precision": 0.7396615073575851,
    "recall": 0.6323145512588655,
    "f1-score": 0.6626992101908417,
    "support": 254356.0
  }
}
```

## Artifacts
- Model: `loan_default_model.zip`
- Metrics: `metrics.json`
- Confusion: `confusion.json`
- History CSV: `history.csv`
- ROC points CSV: `roc_points.csv`
- PR points CSV: `pr_points.csv`
- Learning curves: `figures/learning_curves.png`
- ROC curve: `figures/roc_curve.png`
- PR curve: `figures/pr_curve.png`
- Resolved config: `config_resolved.yaml`
- Features manifest: `features.json`
- H2O leaderboard: `h2o_leaderboard.csv`
- Per-family feature importance: `figures/comparison/per_family_varimp/`, CSVs under `varimp_per_family/`
- Leader partial dependence: `figures/explanations/partial_dependence/`, CSVs under `partial_dependence/`

## Notes
- Evaluated defaults as the positive class.
- Threshold selected according to configured strategy and annotated on curves.