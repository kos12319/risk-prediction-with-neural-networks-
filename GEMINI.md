# GEMINI.md: AI Agent Instructions

This document provides essential context for AI agents interacting with this project. For a complete human-oriented guide, refer to `README.md`.

## Project Overview

This is a Python-based machine learning project for building a neural network-based credit risk model on the LendingClub dataset. The primary goal is to predict loan defaults while also supporting feature subset selection to find a compact, high-value set of predictors.

**Key Technologies:**
*   **Backends:** PyTorch (neural net) and H2O AutoML
*   **Data Handling:** pandas, scikit-learn, imbalanced-learn
*   **Configuration:** YAML (`configs/`)
*   **Experiment Tracking:** Weights & Biases (`wandb`)
*   **Dependency Management:** pip-tools (`requirements.in`, `requirements.txt`)
*   **Orchestration:** Makefile

**Architecture:**
The project is structured as a configurable pipeline driven by a command-line interface (CLI). All operations are initiated via `make` targets, which call Python modules in `src/cli/`. The core logic is separated into modules for data loading (`src/data`), feature preprocessing (`src/features`), model definition (`src/models`), and training (`src/training`). Backends are decoupled: backend-specific orchestration lives under `src/training/backends/<backend>/` and owns concerns like run naming and extra artifacts; shared utilities in `src/training/base_pipeline.py` must remain backend-agnostic. Backends consume the stable interfaces in `src/training/interfaces.py` (not `base_pipeline`) so future internal refactors won’t break backend imports. All artifacts for a given run (model, metrics, figures, logs) are saved to a unique, timestamped directory under `local_runs/`. Run catalog helpers index these runs and produce a Markdown index with ΔAUC vs previous run in each group and a small AUC trend plot per group (`local_runs/index_plots/`), when `matplotlib` is available.

## Building and Running

All workflows should be executed via the `Makefile` to ensure reproducible environments and consistent execution.

**1. Setup:**
First, create the virtual environment and install dependencies.
```bash
make venv
```

**2. PyTorch Training:**
Run the neural-network pipeline using a configuration with `model.backend: pytorch`.
```bash
# Train with the default configuration
make train CONFIG=configs/pytorch_default.yaml

# Train using a different configuration
make train CONFIG=configs/provider_aware.yaml

# Add run notes for tracking
make train CONFIG=configs/pytorch_default.yaml NOTES="testing new dropout"
```

**H2O AutoML:**
Switch to the AutoML backend via `make automl-h2o` (the backend is implied; `model.backend` is optional when using the H2O CLI).
```bash
make automl-h2o
make automl-h2o AUTOML_CONFIG=configs/h2o_default.yaml NOTES="smoke test"
```
- Ensure a Java runtime is available (`java -version`). The CLI performs a pre-flight check and exits with guidance if Java is missing or blocked.

**3. Feature Selection:**
Run feature selection using either Mutual Information (`mi`) or L1 regularization (`l1`).
```bash
make select CONFIG=configs/pytorch_default.yaml METHOD=mi
```

**4. Dry Run:**
Perform an end-to-end check without saving any artifacts. Use the backend-specific target:
```bash
make dryrun CONFIG=configs/pytorch_default.yaml          # PyTorch
make dryrun-h2o AUTOML_CONFIG=configs/h2o_default.yaml   # H2O AutoML
make dryrun-cv                                           # PyTorch temporal CV (2 folds)
make dryrun-h2o-cv                                       # H2O temporal CV (2 folds; Java required)
make cv-train                                            # PyTorch temporal CV then full training (smoke)
```

**5. Experiment Tracking (W&B):**
*   Log in to Weights & Biases (requires `WANDB_API_KEY` env var).
    ```bash
    make wandb-login
    ```
*   Pull run data from W&B to a local directory (`wandb-history/`).
    ```bash
    # Pull a specific run
    make pull-run RUN=<entity/project/run_id>

    # Pull all runs for the configured project
    make pull-all
    ```

## Development Conventions

*   **Makefile-First Policy:** ALWAYS use `make` for running tasks. Do not call `python -m src.cli...` directly. The unified `src.cli train|dryrun` entries are deprecated; use backend-specific CLIs if you must call modules directly.
*   **Configuration:** All parameters (data paths, model hyperparameters, features) are managed via YAML files in `configs/` (`configs/pytorch/` for PyTorch bases, `configs/h2o/` for AutoML). Do not hardcode paths or parameters in scripts.
*   **Data Splitting:** The default and required method for test sets is a time-based split on the `issue_d` column to prevent lookahead bias. Validation sets are carved from the training data *before* oversampling.
*   **Leakage Prevention:** A strict leakage policy is enforced. Only features available at the time of loan origination are used. A list of known leaky columns is maintained in the configuration and dropped automatically.
*   **Dependencies:** Manage dependencies by editing `requirements.in` and running `make deps-compile` to regenerate the `requirements.txt` lockfile.
*   **Testing:** Tests are written with `pytest` and located in the `tests/` directory.
*   **Commits & PRs:** Follow conventional commit style (e.g., `feat: ...`, `fix: ...`). Pull requests should explain the "what" and "why" and include relevant metrics or figures.
*   **Medium backlog:** Config guardrails and temporal CV are implemented; the run catalog is available via `make run-catalog` and a Markdown summary via `make run-catalog-report` (see `docs/architecture/PAIN_POINTS.md`).
