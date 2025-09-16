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
	@echo "  train          Run training (CONFIG=path, default configs/default.yaml)"
	@echo "  select         Run feature selection (CONFIG=..., METHOD=mi|l1)"
	@echo "  dict           Generate column dictionary (CONFIG=..., CSV optional)"
	@echo "  dryrun         Run training as a dry run (no artifacts persisted)"
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
	$(PY) -m src.cli.train --config $(CONFIG)

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

clean-venv:
	rm -rf $(VENV)
