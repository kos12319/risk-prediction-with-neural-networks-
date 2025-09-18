# Simple project automation

VENV := .venv
PYTHON_BIN := $(shell (command -v python3.12 >/dev/null 2>&1 && echo python3.12) || (command -v python3 >/dev/null 2>&1 && echo python3))
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PIP_COMPILE := $(VENV)/bin/pip-compile
PIP_SYNC := $(VENV)/bin/pip-sync

# Detect architecture to avoid forcing OPENBLAS_CORETYPE on non-ARM Macs (causes OMP SHM errors)
MACHINE := $(shell uname -m)
SAFE_BASE := OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL KMP_INIT_AT_FORK=FALSE KMP_DUPLICATE_LIB_OK=TRUE OMP_PROC_BIND=FALSE MPLBACKEND=Agg XDG_CACHE_HOME=.cache MPLCONFIGDIR=.mplcache
ifeq ($(MACHINE),arm64)
  SAFE_ENV := $(SAFE_BASE) OPENBLAS_CORETYPE=ARMV8
else ifeq ($(MACHINE),aarch64)
  SAFE_ENV := $(SAFE_BASE) OPENBLAS_CORETYPE=ARMV8
else
  SAFE_ENV := $(SAFE_BASE)
endif

.PHONY: help venv install train select clean clean-venv deps-tools deps-compile deps-sync \
	clean-cloud-history clean-wandb-local clean-local-history clean-local-runs clean-all-local

help:
	@echo "Targets:"
	@echo "  venv           Create .venv with Python (prefers python3.12) and install requirements"
	@echo "  install        Alias for venv"
	@echo "  train          Run training (CONFIG=path, NOTES=\"what changed\", PULL=true to download W&B files)"
	@echo "  cpu-train      Run training on CPU with minimal threads (CONFIG=..., PULL=true)"
	@echo "  select         Run feature selection (CONFIG=..., METHOD=mi|l1)"
	@echo "  dict           Generate column dictionary (CONFIG=..., CSV optional)"
	@echo "  explore        Explore dataset (CONFIG=..., CSV=path optional)"
	@echo "  dryrun         Run training as a dry run (no artifacts persisted)"
	@echo "  wandb-login    Login to W&B using env (WANDB_API_KEY, WANDB_ENTITY)"
	@echo "  pull-run       Download a W&B run into ./wandb-history/<run_id> (RUN=entity/project/run_id | project/run_id | run_id)"
	@echo "  pull-all       Download all W&B runs into ./wandb-history/<run_id> (ENTITY/PROJECT from env/config)"
	@echo "  clean-cloud-history Delete all W&B runs (and logged artifacts) for project (ENTITY/PROJECT from env/config; FORCE=1)"
	@echo "  clean-local-runs Remove local run folders (local_runs/)"
	@echo "  clean-wandb-local Remove local W&B folder (./wandb)"
	@echo "  clean-local-history Remove W&B history folder (./wandb-history)"
	@echo "  clean-all-local  Remove local_runs/, ./wandb, and ./wandb-history"
	@echo "  clean-venv     Remove the .venv folder"
	@echo "  deps-tools     Install pip-tools into the venv"
	@echo "  deps-compile   Compile requirements.in -> requirements.txt (pinned)"
	@echo "  deps-sync      Sync venv to requirements.txt (exact)"

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
	$(SAFE_ENV) $(PY) -m src.cli.train --config $(CONFIG) $(if $(NOTES),--notes "$(NOTES)",) $(if $(PULL),--pull,)

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

# Usage: make explore CONFIG=configs/default.yaml [CSV=data/raw/full/thesis_data_full.csv]
CSV ?=
explore: venv
	$(SAFE_ENV) $(PY) -m src.cli.explore --config $(CONFIG) $(if $(CSV),--csv $(CSV),)

# Usage: make dryrun CONFIG=configs/default.yaml
dryrun: venv
	$(SAFE_ENV) $(PY) -m src.cli.dryrun --config $(CONFIG)

# W&B helpers
wandb-login: venv
	$(PY) -m src.cli.wandb_login

# Usage: make pull-run RUN=entity/project/run_id | project/run_id | run_id [CONFIG=...]
RUN ?=
FORCE ?=
pull-run: venv
	@if [ -z "$(RUN)" ]; then echo "Set RUN=entity/project/run_id | project/run_id | run_id (use pull-all to sync a project)"; exit 1; fi
	$(PY) -m src.cli.wandb_pull --run $(RUN) --config $(CONFIG) $(if $(FORCE),--force,)

# Usage: make pull-all [ENTITY=...] [PROJECT=...] [CONFIG=configs/default.yaml]
ENTITY ?=
PROJECT ?=
FORCE ?=
pull-all: venv
	@if [ -z "$(ENTITY)" ] && [ -z "$(PROJECT)" ]; then echo "Using ENTITY/PROJECT from env/config if set"; fi
	$(PY) -m src.cli.wandb_pull_all $(if $(ENTITY),--entity $(ENTITY),) $(if $(PROJECT),--project $(PROJECT),) --config $(CONFIG) $(if $(FORCE),--force,)

# Delete all runs (and their logged artifacts) in a project

# Usage: make clean-cloud-history [ENTITY=...] [PROJECT=...] [CONFIG=...] FORCE=1
clean-cloud-history: venv
	@if [ -z "$(FORCE)" ]; then echo "Refusing to delete cloud runs without FORCE=1"; exit 1; fi
	$(PY) -m src.cli.wandb_clean $(if $(ENTITY),--entity $(ENTITY),) $(if $(PROJECT),--project $(PROJECT),) --config $(CONFIG) --yes

clean-venv:
	rm -rf $(VENV)

clean-local-runs:
	rm -rf local_runs

clean-wandb-local:
	rm -rf wandb

clean-local-history:
	rm -rf wandb-history

clean-all-local: clean-local-runs clean-wandb-local clean-local-history


# Dependency management via pip-tools
deps-tools: venv
	$(PIP) install pip-tools

deps-compile: venv
	$(PIP_COMPILE) requirements.in -o requirements.txt

deps-sync: venv
	$(PIP_SYNC) requirements.txt
