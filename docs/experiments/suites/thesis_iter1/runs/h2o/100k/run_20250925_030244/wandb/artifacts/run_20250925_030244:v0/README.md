# Training Summary — run_20250925_030244

Config: `docs/experiments/suites/thesis_iter1/h2o/100k/selected_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.165203

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 345.0 KB |
| Start (UTC) | 2025-09-25T00:02:44+00:00 |
| End (UTC) | 2025-09-25T00:18:28+00:00 |
| Total time | 943.41 s |
| Load | 0.65 s |
| Split | 0.02 s |
| Preprocess | 0.27 s |
| Train | 935.80 s |
| Eval | 6.66 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.725
- Average Precision: 0.427
- Precision (at threshold): 0.324
- Recall (TPR): 0.735
- Specificity (TNR): 0.578
- Confusion: TP=2794, FP=5821, TN=7959, FN=1005
- n_train: 56250
- n_val: 14062
- n_test: 17579
- n_features: 89
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8878848728246319,
    "recall": 0.5775761973875182,
    "f1-score": 0.6998768906085121,
    "support": 13780.0
  },
  "1": {
    "precision": 0.32431804991294255,
    "recall": 0.7354566991313504,
    "f1-score": 0.45013694216207506,
    "support": 3799.0
  },
  "accuracy": 0.611695773365948,
  "macro avg": {
    "precision": 0.6061014613687872,
    "recall": 0.6565164482594343,
    "f1-score": 0.5750069163852936,
    "support": 17579.0
  },
  "weighted avg": {
    "precision": 0.7660923726686784,
    "recall": 0.611695773365948,
    "f1-score": 0.6459055575322271,
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