# Training Summary — run_20250916_192400

Config: `configs/weighted.yaml`
Backend: pytorch
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.539776

## Metrics
- ROC AUC: 0.732
- Average Precision: 0.424
- Precision (at threshold): 0.357
- Recall (TPR): 0.613
- Specificity (TNR): 0.731
- Confusion: TP=214, FP=385, TN=1047, FN=135
- n_train: 7121
- n_test: 1781
- n_features: 115

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.8857868020304569,
    "recall": 0.7311452513966481,
    "f1-score": 0.801071155317521,
    "support": 1432.0
  },
  "1": {
    "precision": 0.3572621035058431,
    "recall": 0.6131805157593123,
    "f1-score": 0.45147679324894513,
    "support": 349.0
  },
  "accuracy": 0.708029197080292,
  "macro avg": {
    "precision": 0.62152445276815,
    "recall": 0.6721628835779803,
    "f1-score": 0.6262739742832331,
    "support": 1781.0
  },
  "weighted avg": {
    "precision": 0.7822185146721805,
    "recall": 0.708029197080292,
    "f1-score": 0.7325655784719662,
    "support": 1781.0
  }
}
```

## Artifacts
- Model: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192400/loan_default_model.pt`
- Metrics: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192400/metrics.json`
- Confusion: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192400/confusion.json`
- History CSV: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192400/history.csv`
- ROC points CSV: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192400/roc_points.csv`
- PR points CSV: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192400/pr_points.csv`
- Learning curves: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192400/figures/learning_curves.png`
- ROC curve: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192400/figures/roc_curve.png`
- PR curve: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192400/figures/pr_curve.png`
- Resolved config: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192400/config_resolved.yaml`
- Features manifest: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_192400/features.json`

## Notes
- Evaluated defaults as the positive class.
- Threshold selected according to configured strategy and annotated on curves.