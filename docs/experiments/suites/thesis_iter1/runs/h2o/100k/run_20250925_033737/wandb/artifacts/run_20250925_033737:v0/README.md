# Training Summary — run_20250925_033737

Config: `docs/experiments/suites/thesis_iter1/h2o/100k/selected_plus_providers_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.192222

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 178.2 KB |
| Start (UTC) | 2025-09-25T00:37:37+00:00 |
| End (UTC) | 2025-09-25T00:53:15+00:00 |
| Total time | 938.55 s |
| Load | 0.58 s |
| Split | 0.02 s |
| Preprocess | 0.22 s |
| Train | 930.99 s |
| Eval | 6.74 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.739
- Average Precision: 0.445
- Precision (at threshold): 0.359
- Recall (TPR): 0.686
- Specificity (TNR): 0.662
- Confusion: TP=2608, FP=4664, TN=9116, FN=1191
- n_train: 56250
- n_val: 14062
- n_test: 17579
- n_features: 133
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8844474628892985,
    "recall": 0.6615384615384615,
    "f1-score": 0.7569228214389505,
    "support": 13780.0
  },
  "1": {
    "precision": 0.3586358635863586,
    "recall": 0.6864964464332719,
    "f1-score": 0.47114081835425886,
    "support": 3799.0
  },
  "accuracy": 0.6669321349337277,
  "macro avg": {
    "precision": 0.6215416632378286,
    "recall": 0.6740174539858668,
    "f1-score": 0.6140318198966047,
    "support": 17579.0
  },
  "weighted avg": {
    "precision": 0.7708142490687246,
    "recall": 0.6669321349337277,
    "f1-score": 0.6951624351986215,
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