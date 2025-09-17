# Simple project automation

VENV := .venv
PYTHON_BIN := $(shell (command -v python3.12 >/dev/null 2>&1 && echo python3.12) || (command -v python3 >/dev/null 2>&1 && echo python3))
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install train select clean clean-venv

help:
	@echo "Targets:"
	@echo "  venv           Create .venv with Python (prefers python3.12) and install requirements"
	@echo "  install        Alias for venv"
	@echo "  train          Run training (CONFIG=path, NOTES=\"what changed\", PULL=true to download W&B files)"
	@echo "  cpu-train      Run training on CPU with minimal threads (CONFIG=..., PULL=true)"
	@echo "  select         Run feature selection (CONFIG=..., METHOD=mi|l1)"
	@echo "  dict           Generate column dictionary (CONFIG=..., CSV optional)"
	@echo "  dryrun         Run training as a dry run (no artifacts persisted)"
	@echo "  wandb-login    Login to W&B using env (WANDB_API_KEY, WANDB_ENTITY)"
	@echo "  pull-run       Download W&B run files/artifacts (RUN=entity/project/run_id | project/run_id | run_id)"
	@echo "  pull-all       Download all W&B runs for a project (ENTITY=... PROJECT=... [TARGET_ROOT=...] [CONFIG=...])"
	@echo "  clean-artifacts Remove local artifact folders (reports/, models/, local_runs/)"
	@echo "  clean-venv     Remove the .venv folder"

$(VENV)/bin/activate: requirements.txt
	@echo "Using Python: $(PYTHON_BIN)"
	$(PYTHON_BIN) -m venv $(VENV)
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

venv: $(VENV)/bin/activate

install: venv

# Usage: make train CONFIG=configs/default.yaml
CONFIG ?= configs/default.yaml
train: venv
	$(PY) -m src.cli.train --config $(CONFIG) $(if $(NOTES),--notes "$(NOTES)",) $(if $(PULL),--pull,)

# CPU-only training helper (good for Linux/WSL/CI)
cpu-train: venv
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	OPENBLAS_NUM_THREADS=1 \
	NUMEXPR_NUM_THREADS=1 \
	VECLIB_MAXIMUM_THREADS=1 \
	BLIS_NUM_THREADS=1 \
	CUDA_VISIBLE_DEVICES= \
	MPLBACKEND=Agg \
	$(PY) -m src.cli.train --config $(CONFIG) --cpu $(if $(NOTES),--notes "$(NOTES)",) $(if $(PULL),--pull,)

# Usage: make select CONFIG=configs/default.yaml METHOD=mi
METHOD ?= mi
select: venv
	$(PY) -m src.cli.select --config $(CONFIG) --method $(METHOD)

# Usage: make dict CONFIG=configs/default.yaml CSV=data/raw/samples/first_10k_rows.csv
CSV ?=
dict: venv
	$(PY) -m src.cli.gen_column_dict --config $(CONFIG) $(if $(CSV),--csv $(CSV),)

# Usage: make dryrun CONFIG=configs/default.yaml
dryrun: venv
	$(PY) -m src.cli.dryrun --config $(CONFIG)

# W&B helpers
wandb-login: venv
	$(PY) -m src.cli.wandb_login

# Usage: make pull-run RUN=entity/project/run_id [CONFIG=configs/default.yaml] [TARGET=dir]
RUN ?=
TARGET ?=
pull-run: venv
	@if [ -z "$(RUN)" ]; then echo "Set RUN=entity/project/run_id | project/run_id | run_id"; exit 1; fi
	$(PY) -m src.cli.wandb_pull --run $(RUN) --config $(CONFIG) $(if $(TARGET),--target $(TARGET),)

# Usage: make pull-all ENTITY=your_entity PROJECT=loan-risk-mlp [TARGET_ROOT=dir] [CONFIG=configs/default.yaml]
ENTITY ?=
PROJECT ?=
TARGET_ROOT ?=
pull-all: venv
	@if [ -z "$(ENTITY)" ] && [ -z "$(PROJECT)" ]; then echo "Set ENTITY and PROJECT or rely on env/config"; fi
	$(PY) -m src.cli.wandb_pull_all $(if $(ENTITY),--entity $(ENTITY),) $(if $(PROJECT),--project $(PROJECT),) $(if $(TARGET_ROOT),--target-root $(TARGET_ROOT),) --config $(CONFIG)

clean-venv:
	rm -rf $(VENV)

clean-artifacts:
	rm -rf reports models local_runs
