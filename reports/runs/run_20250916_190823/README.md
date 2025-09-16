# Training Summary — run_20250916_190823

Config: `configs/default.yaml`
Backend: pytorch
Positive class: positive=default (Charged Off)
Threshold strategy: youden_j
Chosen threshold: 0.219131

## Metrics
- ROC AUC: 0.674
- Average Precision: 0.349
- Precision (at threshold): 0.335
- Recall (TPR): 0.496
- Specificity (TNR): 0.760
- Confusion: TP=173, FP=344, TN=1088, FN=176
- n_train: 11514
- n_test: 1781
- n_features: 115

## Artifacts
- Model: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_190823/loan_default_model.pt`
- Metrics: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_190823/metrics.json`
- Confusion: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_190823/confusion.json`
- History CSV: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_190823/history.csv`
- Learning curves: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_190823/figures/learning_curves.png`
- ROC curve: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_190823/figures/roc_curve.png`
- PR curve: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_190823/figures/pr_curve.png`
- Resolved config: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_190823/config_resolved.yaml`
- Features manifest: `/mnt/c/Users/kosta/Documents/ptyxiaki_tel/reports/runs/run_20250916_190823/features.json`

## Notes
- Evaluated defaults as the positive class.
- Threshold selected according to configured strategy and annotated on curves.