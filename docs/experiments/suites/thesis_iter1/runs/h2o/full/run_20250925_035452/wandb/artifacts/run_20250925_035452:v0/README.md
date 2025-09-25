# Training Summary — run_20250925_035452

Config: `docs/experiments/suites/thesis_iter1/h2o/full/agnostic_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.164860

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 821.4 KB |
| Start (UTC) | 2025-09-25T00:54:52+00:00 |
| End (UTC) | 2025-09-25T02:30:01+00:00 |
| Total time | 5709.25 s |
| Load | 17.41 s |
| Split | 0.41 s |
| Preprocess | 5.72 s |
| Train | 5668.20 s |
| Eval | 17.51 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.700
- Average Precision: 0.384
- Precision (at threshold): 0.339
- Recall (TPR): 0.641
- Specificity (TNR): 0.646
- Confusion: TP=35986, FP=70139, TN=128114, FN=20117
- n_train: 813938
- n_val: 203485
- n_test: 254356
- n_features: 122
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8642861479717469,
    "recall": 0.6462146852758849,
    "f1-score": 0.7395088950716339,
    "support": 198253.0
  },
  "1": {
    "precision": 0.33909069493521793,
    "recall": 0.641427374650197,
    "f1-score": 0.4436472125650319,
    "support": 56103.0
  },
  "accuracy": 0.6451587538725251,
  "macro avg": {
    "precision": 0.6016884214534823,
    "recall": 0.6438210299630409,
    "f1-score": 0.5915780538183328,
    "support": 254356.0
  },
  "weighted avg": {
    "precision": 0.748444412366106,
    "recall": 0.6451587538725251,
    "f1-score": 0.6742510361114841,
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