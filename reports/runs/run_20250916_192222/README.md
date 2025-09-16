# Training Summary — run_20250916_192222

Config: `configs/weighted.yaml`
Backend: pytorch
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.500452

## Metrics
- ROC AUC: 0.728
- Average Precision: 0.429
- Precision (at threshold): 0.356
- Recall (TPR): 0.605
- Specificity (TNR): 0.733
- Confusion: TP=211, FP=382, TN=1050, FN=138
- n_train: 7121
- n_test: 1781
- n_features: 115

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8838383838383839,
    "recall": 0.7332402234636871,
    "f1-score": 0.8015267175572519,
    "support": 1432.0
  },
  "1": {
    "precision": 0.35581787521079256,
    "recall": 0.6045845272206304,
    "f1-score": 0.44798301486199577,
    "support": 349.0
  },
  "accuracy": 0.708029197080292,
  "macro avg": {
    "precision": 0.6198281295245882,
    "recall": 0.6689123753421587,
    "f1-score": 0.6247548662096238,
    "support": 1781.0
  },
  "weighted avg": {
    "precision": 0.7803688961848019,
    "recall": 0.708029197080292,
    "f1-score": 0.732247238477721,
    "support": 1781.0
  }
}
```

## Artifacts
- Model: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192222/loan_default_model.pt`
- Metrics: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192222/metrics.json`
- Confusion: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192222/confusion.json`
- History CSV: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192222/history.csv`
- ROC points CSV: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192222/roc_points.csv`
- PR points CSV: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192222/pr_points.csv`
- Learning curves: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192222/figures/learning_curves.png`
- ROC curve: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192222/figures/roc_curve.png`
- PR curve: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192222/figures/pr_curve.png`
- Resolved config: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192222/config_resolved.yaml`
- Features manifest: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192222/features.json`

## Notes
- Evaluated defaults as the positive class.
- Threshold selected according to configured strategy and annotated on curves.