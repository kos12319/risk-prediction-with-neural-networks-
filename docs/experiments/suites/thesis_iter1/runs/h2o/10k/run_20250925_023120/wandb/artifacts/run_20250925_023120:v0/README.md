# Training Summary — run_20250925_023120

Config: `docs/experiments/suites/thesis_iter1/h2o/10k/aware_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.148700

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 217.5 KB |
| Start (UTC) | 2025-09-24T23:31:20+00:00 |
| End (UTC) | 2025-09-24T23:36:50+00:00 |
| Total time | 330.18 s |
| Load | 0.07 s |
| Split | 0.00 s |
| Preprocess | 0.05 s |
| Train | 324.00 s |
| Eval | 6.06 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.759
- Average Precision: 0.460
- Precision (at threshold): 0.313
- Recall (TPR): 0.771
- Specificity (TNR): 0.587
- Confusion: TP=269, FP=591, TN=841, FN=80
- n_train: 5697
- n_val: 1424
- n_test: 1781
- n_features: 159
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.9131378935939196,
    "recall": 0.5872905027932961,
    "f1-score": 0.7148321291967701,
    "support": 1432.0
  },
  "1": {
    "precision": 0.3127906976744186,
    "recall": 0.7707736389684814,
    "f1-score": 0.44499586435070304,
    "support": 349.0
  },
  "accuracy": 0.6232453677709152,
  "macro avg": {
    "precision": 0.6129642956341691,
    "recall": 0.6790320708808888,
    "f1-score": 0.5799139967737366,
    "support": 1781.0
  },
  "weighted avg": {
    "precision": 0.7954954616029561,
    "recall": 0.6232453677709152,
    "f1-score": 0.6619557359169962,
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