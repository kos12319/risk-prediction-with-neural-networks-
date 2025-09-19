# GEMINI.md: AI Agent Instructions

This document provides essential context for AI agents interacting with this project. For a complete human-oriented guide, refer to `README.md`.

## Project Overview

This is a Python-based machine learning project for building a neural network-based credit risk model on the LendingClub dataset. The primary goal is to predict loan defaults while also supporting feature subset selection to find a compact, high-value set of predictors.

**Key Technologies:**
*   **Backend:** PyTorch
*   **Data Handling:** pandas, scikit-learn, imbalanced-learn
*   **Configuration:** YAML (`configs/`)
*   **Experiment Tracking:** Weights & Biases (`wandb`)
*   **Dependency Management:** pip-tools (`requirements.in`, `requirements.txt`)
*   **Orchestration:** Makefile

**Architecture:**
The project is structured as a configurable pipeline driven by a command-line interface (CLI). All operations are initiated via `make` targets, which call Python modules in `src/cli/`. The core logic is separated into modules for data loading (`src/data`), feature preprocessing (`src/features`), model definition (`src/models`), and training (`src/training`). All artifacts for a given run (model, metrics, figures, logs) are saved to a unique, timestamped directory under `local_runs/`.

## Building and Running

All workflows should be executed via the `Makefile` to ensure reproducible environments and consistent execution.

**1. Setup:**
First, create the virtual environment and install dependencies.
```bash
make venv
```

**2. Training:**
Run the training pipeline using a configuration file.
```bash
# Train with the default configuration
make train CONFIG=configs/default.yaml

# Train using a different configuration
make train CONFIG=configs/provider_aware.yaml

# Add run notes for tracking
make train CONFIG=configs/default.yaml NOTES="testing new dropout"
```

**3. Feature Selection:**
Run feature selection using either Mutual Information (`mi`) or L1 regularization (`l1`).
```bash
make select CONFIG=configs/default.yaml METHOD=mi
```

**4. Dry Run:**
Perform an end-to-end check without saving any artifacts. This is useful for validating a configuration change.
```bash
make dryrun CONFIG=configs/default.yaml
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

*   **Makefile-First Policy:** ALWAYS use `make` for running tasks. Do not call `python -m src.cli...` directly.
*   **Configuration:** All parameters (data paths, model hyperparameters, features) are managed via YAML files in `configs/`. Do not hardcode paths or parameters in scripts.
*   **Data Splitting:** The default and required method for test sets is a time-based split on the `issue_d` column to prevent lookahead bias. Validation sets are carved from the training data *before* oversampling.
*   **Leakage Prevention:** A strict leakage policy is enforced. Only features available at the time of loan origination are used. A list of known leaky columns is maintained in the configuration and dropped automatically.
*   **Dependencies:** Manage dependencies by editing `requirements.in` and running `make deps-compile` to regenerate the `requirements.txt` lockfile.
*   **Testing:** Tests are written with `pytest` and located in the `tests/` directory.
*   **Commits & PRs:** Follow conventional commit style (e.g., `feat: ...`, `fix: ...`). Pull requests should explain the "what" and "why" and include relevant metrics or figures.
