# Training Summary — run_20250916_185608

Config: `configs/default.yaml`
Backend: pytorch
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.099802

## Metrics
- ROC AUC: 0.676
- Average Precision: 0.360
- Precision (at threshold): 0.322
- Recall (TPR): 0.610
- Specificity (TNR): 0.686
- Confusion: TP=213, FP=449, TN=983, FN=136
- n_train: 11514
- n_test: 1781
- n_features: 115

## Artifacts
- Model: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/models/loan_default_model.pt`
- Metrics: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/metrics.json`
- Confusion: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/confusion.json`
- Learning curves: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/figures/learning_curves.png`
- ROC curve: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/figures/roc_curve.png`
- PR curve: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/figures/pr_curve.png`

## Notes
- Evaluated defaults as the positive class.
- Threshold selected according to configured strategy and annotated on curves.