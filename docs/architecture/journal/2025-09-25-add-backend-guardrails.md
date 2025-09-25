# Add backend guardrails

- Date: 2025-09-25
- Status: done
- Tags: training, h2o, cli

## Summary
Separated backend orchestration into dedicated adapters, taught the selection CLI to reuse the shared config loader, and added an H2O Java pre-flight so runs fail fast with actionable messaging in sandboxed environments. Updated docs to reflect the split CLIs/configs and clarified Java requirements.

## Impact
- configs: configs/pytorch/base.yaml, configs/h2o/base.yaml, configs/pytorch_default.yaml, configs/h2o_default.yaml
- Make: make dryrun, make dryrun-h2o, make select, make docs-journal-new (fails under current sandbox)
- code: src/training/base_pipeline.py, src/training/backends/pytorch/pipeline.py, src/training/backends/h2o/pipeline.py, src/training/train_h2o.py, src/training/config.py, src/cli/train.py, src/cli/automl_h2o.py, src/cli/dryrun.py, src/cli/dryrun_h2o.py, src/cli/select.py, tests/test_training_backends.py, tests/test_training_config.py, pytest.ini

## Next
- Track medium-level backlog items in docs/PAIN_POINTS.md and revisit H2O support once a Java-capable environment is available.
