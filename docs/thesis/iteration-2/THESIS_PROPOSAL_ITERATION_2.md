# Thesis Proposal — Iteration 2: Neural-Centric Refinement with Temporal CV

## Objective
- Build a strong, reproducible neural-network baseline for LendingClub default prediction that competes with tree ensembles, using a time-aware evaluation and calibration. Compare against H2O AutoML winners and document where NNs win/lose across dataset scales.

## Hypotheses
- With categorical embeddings, monotonic priors on known risk drivers, and strong regularization, an MLP can match or exceed AutoML tree ensembles on small-to-medium samples (1k–10k) and narrow the gap on larger ones (100k+).
- Temporal CV plus validation-chosen thresholds improve stability and out-of-sample calibration vs. single split.

## Methods
- Backend design (Makefile-first):
  - PyTorch MLP: embeddings for `grade/sub_grade/purpose/term`, BatchNorm, dropout, weight decay, cyclic/one-cycle LR, early stopping.
  - Monotonic cues: soft monotonic regularization for features like `int_rate`, `fico_avg` (non-decreasing / non-increasing as appropriate).
  - Class imbalance: balanced mini-batches or focal loss; monitor AUCPR and calibration.
  - Calibration: Platt/Isotonic on the validation fold; apply fixed threshold to test.
  - Baselines: H2O AutoML winners from Iteration 1 for each size/feature regime.
- Evaluation invariants:
  - Time split by `issue_d`; validation carved from train only; oversample train subset only.
  - Threshold selected on validation via configured strategy (`fixed|youden_j|f1`), fixed on test; respect `eval.pos_label=0`.
  - Determinism: seed Python/NumPy/Torch/DataLoader workers; headless plotting and thread limits via Makefile.
- Temporal CV:
  - Expanding-window k-fold (`split.cv`), aggregate metrics to `reports/cv_metrics.json` with `train_full_after: true` for final refit.

## Experiments
- Thesis snapshot: `docs/thesis/iteration-2/` (iteration narrative + key figures)
- Suite reference: `docs/experiments/suites/thesis_iter1/` (reports reused for baselines)
- Data scales: 1k, 10k, 100k, full.
- Feature regimes: replicate Iteration 1 (agnostic, selected, aware, selected+providers) for fair comparison where feasible.
- Commands (Makefile-first examples):
  - PyTorch smoke: `make dryrun`
  - Temporal CV (small): `make dryrun-cv`
  - H2O baseline (size/config): `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/thesis_iter1/h2o/10k/aware.yaml`
  - Full PyTorch train (example): `make train CONFIG=configs/pytorch/base.yaml OVERRIDES="split.method=time,split.cv=5,train_full_after=true"`

## Success Criteria
- 1k/10k: NN AUCPR ≥ AutoML within 1–2% absolute; well-calibrated probabilities at the validation-chosen threshold.
- 100k/full: NN narrows AUCPR gap vs. AutoML; calibration error decreases with temporal CV.
- Clear, reproducible artifacts: config snapshots, metrics.json, PR/ROC curves, per-fold reports.

## Risks & Mitigations
- Overfitting on small samples → strong regularization, early stopping, and sanity checks via `make dryrun`.
- Categorical cardinality/encoding pitfalls → embeddings with dropout and cautious dimensioning; avoid brittle OHE mapping.
- Compute limits on 100k/full → batch size/num workers tuned; prefer `.toarray()` over `.todense()` when converting sparse.

## Artefacts & Reporting
- Iterate NN configs under `configs/pytorch/`; keep runs reproducible via Makefile and seeded generators.
- Compare against H2O leaders; export figures and tables under `docs/thesis/iteration-2/`.
- Document thresholding decisions and operating points alongside AUCPR/ROC.

