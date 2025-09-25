# Training Summary — run_20250925_070714

Config: `docs/experiments/suites/thesis_iter1/h2o/full/aware_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.176486

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 649.9 KB |
| Start (UTC) | 2025-09-25T04:07:14+00:00 |
| End (UTC) | 2025-09-25T05:42:26+00:00 |
| Total time | 5711.98 s |
| Load | 17.51 s |
| Split | 0.46 s |
| Preprocess | 7.40 s |
| Train | 5669.18 s |
| Eval | 17.43 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.709
- Average Precision: 0.393
- Precision (at threshold): 0.347
- Recall (TPR): 0.646
- Specificity (TNR): 0.656
- Confusion: TP=36227, FP=68284, TN=129969, FN=19876
- n_train: 813938
- n_val: 203485
- n_test: 254356
- n_features: 166
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8673562681437486,
    "recall": 0.6555714163215689,
    "f1-score": 0.7467379875782107,
    "support": 198253.0
  },
  "1": {
    "precision": 0.3466333687363053,
    "recall": 0.645723045113452,
    "f1-score": 0.4511063792695531,
    "support": 56103.0
  },
  "accuracy": 0.6533991728129079,
  "macro avg": {
    "precision": 0.6069948184400269,
    "recall": 0.6506472307175104,
    "f1-score": 0.5989221834238819,
    "support": 254356.0
  },
  "weighted avg": {
    "precision": 0.7525010383655802,
    "recall": 0.6533991728129079,
    "f1-score": 0.6815308758098992,
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