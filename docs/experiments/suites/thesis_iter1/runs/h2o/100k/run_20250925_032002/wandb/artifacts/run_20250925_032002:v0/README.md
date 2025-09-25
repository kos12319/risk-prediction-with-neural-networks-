# Training Summary — run_20250925_032002

Config: `docs/experiments/suites/thesis_iter1/h2o/100k/aware_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.178280

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 303.0 KB |
| Start (UTC) | 2025-09-25T00:20:02+00:00 |
| End (UTC) | 2025-09-25T00:35:54+00:00 |
| Total time | 951.61 s |
| Load | 0.82 s |
| Split | 0.03 s |
| Preprocess | 0.48 s |
| Train | 943.54 s |
| Eval | 6.73 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.743
- Average Precision: 0.452
- Precision (at threshold): 0.349
- Recall (TPR): 0.707
- Specificity (TNR): 0.636
- Confusion: TP=2687, FP=5015, TN=8765, FN=1112
- n_train: 56250
- n_val: 14062
- n_test: 17579
- n_features: 159
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8874152070466741,
    "recall": 0.636066763425254,
    "f1-score": 0.7410068901382255,
    "support": 13780.0
  },
  "1": {
    "precision": 0.348870423266684,
    "recall": 0.707291392471703,
    "f1-score": 0.4672637161985914,
    "support": 3799.0
  },
  "accuracy": 0.6514591273678821,
  "macro avg": {
    "precision": 0.618142815156679,
    "recall": 0.6716790779484785,
    "f1-score": 0.6041353031684085,
    "support": 17579.0
  },
  "weighted avg": {
    "precision": 0.771030223055538,
    "recall": 0.6514591273678821,
    "f1-score": 0.6818482168464188,
    "support": 17579.0
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