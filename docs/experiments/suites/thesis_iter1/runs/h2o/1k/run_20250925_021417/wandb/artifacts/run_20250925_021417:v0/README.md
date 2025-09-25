# Training Summary — run_20250925_021417

Config: `docs/experiments/suites/thesis_iter1/h2o/1k/selected_plus_providers_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.338929

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 278.3 KB |
| Start (UTC) | 2025-09-24T23:14:17+00:00 |
| End (UTC) | 2025-09-24T23:15:43+00:00 |
| Total time | 85.42 s |
| Load | 0.01 s |
| Split | 0.00 s |
| Preprocess | 0.01 s |
| Train | 79.64 s |
| Eval | 5.75 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.649
- Average Precision: 0.254
- Precision (at threshold): 0.240
- Recall (TPR): 0.231
- Specificity (TNR): 0.872
- Confusion: TP=6, FP=19, TN=130, FN=20
- n_train: 558
- n_val: 140
- n_test: 175
- n_features: 122
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8666666666666667,
    "recall": 0.87248322147651,
    "f1-score": 0.8695652173913043,
    "support": 149.0
  },
  "1": {
    "precision": 0.24,
    "recall": 0.23076923076923078,
    "f1-score": 0.23529411764705882,
    "support": 26.0
  },
  "accuracy": 0.7771428571428571,
  "macro avg": {
    "precision": 0.5533333333333333,
    "recall": 0.5516262261228704,
    "f1-score": 0.5524296675191815,
    "support": 175.0
  },
  "weighted avg": {
    "precision": 0.7735619047619048,
    "recall": 0.7771428571428571,
    "f1-score": 0.7753306540007308,
    "support": 175.0
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