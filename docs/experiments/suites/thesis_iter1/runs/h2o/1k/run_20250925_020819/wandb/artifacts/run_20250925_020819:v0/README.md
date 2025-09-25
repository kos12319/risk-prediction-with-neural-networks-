# Training Summary — run_20250925_020819

Config: `docs/experiments/suites/thesis_iter1/h2o/1k/selected_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.140978

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 136.6 KB |
| Start (UTC) | 2025-09-24T23:08:19+00:00 |
| End (UTC) | 2025-09-24T23:09:47+00:00 |
| Total time | 87.34 s |
| Load | 0.01 s |
| Split | 0.00 s |
| Preprocess | 0.01 s |
| Train | 81.43 s |
| Eval | 5.89 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.649
- Average Precision: 0.238
- Precision (at threshold): 0.247
- Recall (TPR): 0.692
- Specificity (TNR): 0.631
- Confusion: TP=18, FP=55, TN=94, FN=8
- n_train: 558
- n_val: 140
- n_test: 175
- n_features: 83
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.9215686274509803,
    "recall": 0.6308724832214765,
    "f1-score": 0.749003984063745,
    "support": 149.0
  },
  "1": {
    "precision": 0.2465753424657534,
    "recall": 0.6923076923076923,
    "f1-score": 0.36363636363636365,
    "support": 26.0
  },
  "accuracy": 0.64,
  "macro avg": {
    "precision": 0.5840719849583669,
    "recall": 0.6615900877645844,
    "f1-score": 0.5563201738500543,
    "support": 175.0
  },
  "weighted avg": {
    "precision": 0.8212839108246037,
    "recall": 0.64,
    "f1-score": 0.6917493661716769,
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