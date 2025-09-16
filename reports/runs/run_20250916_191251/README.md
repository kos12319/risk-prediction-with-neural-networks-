# Training Summary — run_20250916_191251

Config: `configs/default.yaml`
Backend: pytorch
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.434388

## Metrics
- ROC AUC: 0.671
- Average Precision: 0.366
- Precision (at threshold): 0.355
- Recall (TPR): 0.476
- Specificity (TNR): 0.790
- Confusion: TP=166, FP=301, TN=1131, FN=183
- n_train: 11514
- n_test: 1781
- n_features: 115

## Classification Report (at threshold)
```json
{
  "0": {
    "precision": 0.860730593607306,
    "recall": 0.789804469273743,
    "f1-score": 0.8237436270939549,
    "support": 1432.0
  },
  "1": {
    "precision": 0.3554603854389722,
    "recall": 0.47564469914040114,
    "f1-score": 0.4068627450980392,
    "support": 349.0
  },
  "accuracy": 0.7282425603593486,
  "macro avg": {
    "precision": 0.6080954895231391,
    "recall": 0.6327245842070721,
    "f1-score": 0.615303186095997,
    "support": 1781.0
  },
  "weighted avg": {
    "precision": 0.7617191940279975,
    "recall": 0.7282425603593486,
    "f1-score": 0.7420527636371471,
    "support": 1781.0
  }
}
```

## Artifacts
- Model: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_191251/loan_default_model.pt`
- Metrics: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_191251/metrics.json`
- Confusion: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_191251/confusion.json`
- History CSV: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_191251/history.csv`
- Learning curves: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_191251/figures/learning_curves.png`
- ROC curve: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_191251/figures/roc_curve.png`
- PR curve: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_191251/figures/pr_curve.png`
- Resolved config: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_191251/config_resolved.yaml`
- Features manifest: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_191251/features.json`

## Notes
- Evaluated defaults as the positive class.
- Threshold selected according to configured strategy and annotated on curves.