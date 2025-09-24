Experiment suite: LendingClub full dataset — feature subsets and splits

This suite runs the full dataset across PyTorch and H2O backends, comparing:
- Provider-agnostic vs provider-aware features
- Feature-selection subsets (MI, L1, L1-on-MI)
- Time-based split with temporal CV vs random split

How to run
- Ensure the full CSV exists at `data/raw/full/thesis_data_full.csv`.
- Install deps and create venv: `make install`
- Optional (W&B): `make wandb-login` (set env `WANDB_API_KEY`).
- From this folder, run one of:
  - Foreground: `bash run_experiments.sh`
  - Background: `nohup bash run_experiments.sh > _logs/runner.out 2>&1 &`
  - Monitor logs: `tail -f _logs/*.log`

Run from repository root (alternative)
- `bash docs/experiments/suites/lc_full_feature_sweep/run_experiments.sh`

Run count (trimmed)
- Total: 16 runs
- PyTorch: 8 (agnostic/aware time + CV5 + random; L1 time; MI time)
- H2O: 8 (agnostic/aware time + random; L1 time+random; MI time+random)

Notes
- Configs use `extends` to inherit defaults from `configs/` and only override what differs.
- Temporal CV writes `reports/cv_metrics.json` and per-fold artifacts under `local_runs/.../folds/`.
- H2O configs set `automl.nthreads` and `automl.max_mem_size` for good local performance.

Manual single-run examples (optional)
- PyTorch: `make train CONFIG=docs/experiments/suites/lc_full_feature_sweep/pytorch/agnostic_time.yaml`
- H2O: `make automl-h2o AUTOML_CONFIG=docs/experiments/suites/lc_full_feature_sweep/h2o/agnostic_time.yaml`
