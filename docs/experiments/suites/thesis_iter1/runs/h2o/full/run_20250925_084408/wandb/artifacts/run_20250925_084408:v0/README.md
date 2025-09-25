# Training Summary — run_20250925_084408

Config: `docs/experiments/suites/thesis_iter1/h2o/full/selected_plus_providers_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.164387

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 393.2 KB |
| Start (UTC) | 2025-09-25T05:44:08+00:00 |
| End (UTC) | 2025-09-25T07:18:45+00:00 |
| Total time | 5677.08 s |
| Load | 12.59 s |
| Split | 0.31 s |
| Preprocess | 3.17 s |
| Train | 5643.09 s |
| Eval | 17.92 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.700
- Average Precision: 0.382
- Precision (at threshold): 0.324
- Recall (TPR): 0.707
- Specificity (TNR): 0.582
- Confusion: TP=39662, FP=82919, TN=115334, FN=16441
- n_train: 813938
- n_val: 203485
- n_test: 254356
- n_features: 140
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8752343008916714,
    "recall": 0.5817516002279915,
    "f1-score": 0.6989346358490794,
    "support": 198253.0
  },
  "1": {
    "precision": 0.32355748443886084,
    "recall": 0.7069497174839136,
    "f1-score": 0.4439345436636744,
    "support": 56103.0
  },
  "accuracy": 0.6093663998490305,
  "macro avg": {
    "precision": 0.5993958926652662,
    "recall": 0.6443506588559526,
    "f1-score": 0.571434589756377,
    "support": 254356.0
  },
  "weighted avg": {
    "precision": 0.7535516024947316,
    "recall": 0.6093663998490305,
    "f1-score": 0.6426895692028128,
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