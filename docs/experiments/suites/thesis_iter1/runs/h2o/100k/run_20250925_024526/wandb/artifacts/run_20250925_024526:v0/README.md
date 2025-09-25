# Training Summary — run_20250925_024526

Config: `docs/experiments/suites/thesis_iter1/h2o/100k/agnostic_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.170947

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 186.2 KB |
| Start (UTC) | 2025-09-24T23:45:26+00:00 |
| End (UTC) | 2025-09-25T00:01:10+00:00 |
| Total time | 943.82 s |
| Load | 0.74 s |
| Split | 0.03 s |
| Preprocess | 0.36 s |
| Train | 936.19 s |
| Eval | 6.50 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.730
- Average Precision: 0.442
- Precision (at threshold): 0.331
- Recall (TPR): 0.726
- Specificity (TNR): 0.595
- Confusion: TP=2757, FP=5579, TN=8201, FN=1042
- n_train: 56250
- n_val: 14062
- n_test: 17579
- n_features: 115
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8872660391647733,
    "recall": 0.5951378809869375,
    "f1-score": 0.7124180167658428,
    "support": 13780.0
  },
  "1": {
    "precision": 0.3307341650671785,
    "recall": 0.7257172940247434,
    "f1-score": 0.4543881334981459,
    "support": 3799.0
  },
  "accuracy": 0.6233574150975596,
  "macro avg": {
    "precision": 0.6090001021159759,
    "recall": 0.6604275875058405,
    "f1-score": 0.5834030751319943,
    "support": 17579.0
  },
  "weighted avg": {
    "precision": 0.7669938627214737,
    "recall": 0.6233574150975596,
    "f1-score": 0.6566551447859816,
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