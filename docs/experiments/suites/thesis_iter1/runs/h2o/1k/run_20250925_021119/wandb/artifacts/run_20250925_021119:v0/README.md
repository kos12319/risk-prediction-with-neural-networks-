# Training Summary — run_20250925_021119

Config: `docs/experiments/suites/thesis_iter1/h2o/1k/aware_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.028612

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 169.8 KB |
| Start (UTC) | 2025-09-24T23:11:19+00:00 |
| End (UTC) | 2025-09-24T23:12:46+00:00 |
| Total time | 86.92 s |
| Load | 0.01 s |
| Split | 0.00 s |
| Preprocess | 0.01 s |
| Train | 81.16 s |
| Eval | 5.73 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.592
- Average Precision: 0.212
- Precision (at threshold): 0.197
- Recall (TPR): 0.500
- Specificity (TNR): 0.644
- Confusion: TP=13, FP=53, TN=96, FN=13
- n_train: 558
- n_val: 140
- n_test: 175
- n_features: 148
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8807339449541285,
    "recall": 0.6442953020134228,
    "f1-score": 0.7441860465116279,
    "support": 149.0
  },
  "1": {
    "precision": 0.19696969696969696,
    "recall": 0.5,
    "f1-score": 0.2826086956521739,
    "support": 26.0
  },
  "accuracy": 0.6228571428571429,
  "macro avg": {
    "precision": 0.5388518209619128,
    "recall": 0.5721476510067114,
    "f1-score": 0.5133973710819009,
    "support": 175.0
  },
  "weighted avg": {
    "precision": 0.779146113825013,
    "recall": 0.6228571428571429,
    "f1-score": 0.6756088400982233,
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