# Training Summary — run_20250925_021716

Config: `docs/experiments/suites/thesis_iter1/h2o/10k/agnostic_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.131493

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 161.1 KB |
| Start (UTC) | 2025-09-24T23:17:16+00:00 |
| End (UTC) | 2025-09-24T23:22:47+00:00 |
| Total time | 330.22 s |
| Load | 0.07 s |
| Split | 0.00 s |
| Preprocess | 0.04 s |
| Train | 324.13 s |
| Eval | 5.98 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.747
- Average Precision: 0.451
- Precision (at threshold): 0.275
- Recall (TPR): 0.862
- Specificity (TNR): 0.446
- Confusion: TP=301, FP=793, TN=639, FN=48
- n_train: 5697
- n_val: 1424
- n_test: 1781
- n_features: 115
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.9301310043668122,
    "recall": 0.44622905027932963,
    "f1-score": 0.6031146767343086,
    "support": 1432.0
  },
  "1": {
    "precision": 0.27513711151736747,
    "recall": 0.8624641833810889,
    "f1-score": 0.4171864171864172,
    "support": 349.0
  },
  "accuracy": 0.527793374508703,
  "macro avg": {
    "precision": 0.6026340579420898,
    "recall": 0.6543466168302092,
    "f1-score": 0.5101505469603629,
    "support": 1781.0
  },
  "weighted avg": {
    "precision": 0.8017801516972691,
    "recall": 0.527793374508703,
    "f1-score": 0.5666806719155472,
    "support": 1781.0
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