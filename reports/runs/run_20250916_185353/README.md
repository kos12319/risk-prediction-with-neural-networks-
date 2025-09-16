# Training Summary — run_20250916_185353

Config: `configs/default.yaml`
Backend: pytorch
Positive class: positive=default (Charged Off)
Threshold strategy: f1
Chosen threshold: 0.079821

## Metrics
- ROC AUC: 0.643
- Average Precision: 0.317
- n_train: 11514
- n_test: 1781
- n_features: 115

## Artifacts
- Model: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/models/loan_default_model.pt`
- Metrics: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/metrics.json`
- Learning curves: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/figures/learning_curves.png`
- ROC curve: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/figures/roc_curve.png`
- PR curve: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/figures/pr_curve.png`

## Notes
- Evaluated defaults as the positive class.
- Threshold selected according to configured strategy and annotated on curves.