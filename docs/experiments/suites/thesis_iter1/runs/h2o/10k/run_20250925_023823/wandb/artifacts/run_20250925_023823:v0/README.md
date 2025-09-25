# Training Summary — run_20250925_023823

Config: `docs/experiments/suites/thesis_iter1/h2o/10k/selected_plus_providers_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.203369

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 203.9 KB |
| Start (UTC) | 2025-09-24T23:38:23+00:00 |
| End (UTC) | 2025-09-24T23:43:53+00:00 |
| Total time | 329.51 s |
| Load | 0.06 s |
| Split | 0.00 s |
| Preprocess | 0.03 s |
| Train | 323.36 s |
| Eval | 6.06 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.752
- Average Precision: 0.426
- Precision (at threshold): 0.377
- Recall (TPR): 0.599
- Specificity (TNR): 0.759
- Confusion: TP=209, FP=345, TN=1087, FN=140
- n_train: 5697
- n_val: 1424
- n_test: 1781
- n_features: 133
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8859005704971475,
    "recall": 0.7590782122905028,
    "f1-score": 0.8176006017299736,
    "support": 1432.0
  },
  "1": {
    "precision": 0.37725631768953066,
    "recall": 0.5988538681948424,
    "f1-score": 0.4629014396456257,
    "support": 349.0
  },
  "accuracy": 0.7276810780460415,
  "macro avg": {
    "precision": 0.6315784440933391,
    "recall": 0.6789660402426726,
    "f1-score": 0.6402510206877996,
    "support": 1781.0
  },
  "weighted avg": {
    "precision": 0.786228002147985,
    "recall": 0.7276810780460415,
    "f1-score": 0.7480947019167015,
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