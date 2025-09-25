# Training Summary — run_20250925_022418

Config: `docs/experiments/suites/thesis_iter1/h2o/10k/selected_time.yaml`
Backend: h2o
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.387908

## Run Summary

| Key | Value |
| --- | --- |
| Device | h2o |
| Epochs (ran) | 0 |
| Param count | n/a |
| Model size | 448.3 KB |
| Start (UTC) | 2025-09-24T23:24:18+00:00 |
| End (UTC) | 2025-09-24T23:29:46+00:00 |
| Total time | 327.85 s |
| Load | 0.06 s |
| Split | 0.00 s |
| Preprocess | 0.03 s |
| Train | 321.78 s |
| Eval | 5.98 s |

## What Changed
thesis_iter1 time-budgeted rerun after extends fix

## Metrics
- ROC AUC: 0.736
- Average Precision: 0.421
- Precision (at threshold): 0.286
- Recall (TPR): 0.779
- Specificity (TNR): 0.527
- Confusion: TP=272, FP=678, TN=754, FN=77
- n_train: 5697
- n_val: 1424
- n_test: 1781
- n_features: 89
- Resampling: disabled

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.9073405535499398,
    "recall": 0.526536312849162,
    "f1-score": 0.6663720724701724,
    "support": 1432.0
  },
  "1": {
    "precision": 0.2863157894736842,
    "recall": 0.7793696275071633,
    "f1-score": 0.41878367975365666,
    "support": 349.0
  },
  "accuracy": 0.5760808534531162,
  "macro avg": {
    "precision": 0.596828171511812,
    "recall": 0.6529529701781627,
    "f1-score": 0.5425778761119145,
    "support": 1781.0
  },
  "weighted avg": {
    "precision": 0.7856462005670014,
    "recall": 0.5760808534531162,
    "f1-score": 0.6178553127520006,
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