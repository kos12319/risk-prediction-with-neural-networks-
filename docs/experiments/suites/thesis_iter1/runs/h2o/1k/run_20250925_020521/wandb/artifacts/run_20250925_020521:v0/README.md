# Training Summary — run_20250925_020521

Config: `docs/experiments/suites/thesis_iter1/h2o/1k/agnostic_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.471159

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 273.7 KB |
| Start (UTC) | 2025-09-24T23:05:21+00:00 |
| End (UTC) | 2025-09-24T23:06:48+00:00 |
| Total time | 87.11 s |
| Load | 0.02 s |
| Split | 0.00 s |
| Preprocess | 0.01 s |
| Train | 81.23 s |
| Eval | 5.85 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.731
- Average Precision: 0.315
- Precision (at threshold): 0.255
- Recall (TPR): 0.462
- Specificity (TNR): 0.765
- Confusion: TP=12, FP=35, TN=114, FN=14
- n_train: 558
- n_val: 140
- n_test: 175
- n_features: 109
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.890625,
    "recall": 0.7651006711409396,
    "f1-score": 0.8231046931407943,
    "support": 149.0
  },
  "1": {
    "precision": 0.2553191489361702,
    "recall": 0.46153846153846156,
    "f1-score": 0.3287671232876712,
    "support": 26.0
  },
  "accuracy": 0.72,
  "macro avg": {
    "precision": 0.5729720744680851,
    "recall": 0.6133195663397006,
    "f1-score": 0.5759359082142328,
    "support": 175.0
  },
  "weighted avg": {
    "precision": 0.7962367021276595,
    "recall": 0.72,
    "f1-score": 0.7496602541911875,
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