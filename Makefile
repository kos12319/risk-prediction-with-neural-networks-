# Simple project automation

VENV := .venv
PYTHON_BIN := $(shell (command -v python3.12 >/dev/null 2>&1 && echo python3.12) || (command -v python3 >/dev/null 2>&1 && echo python3))
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PIP_COMPILE := $(VENV)/bin/pip-compile
PIP_SYNC := $(VENV)/bin/pip-sync
AUTOML_CONFIG ?= configs/h2o_default.yaml
H2O_BALANCE ?= 1
H2O_OVERSAMPLING ?= 0
H2O_MAX_AFTER_BALANCE ?=
H2O_CLASS_SAMPLING_FACTORS ?=

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




.PHONY: help venv install train automl-h2o select clean clean-venv deps-tools deps-compile deps-sync \
	clean-cloud-history clean-wandb-local clean-local-history clean-local-runs clean-selection-runs clean-all-local \
	marker-install marker-pdf docs docs-canon docs-journal-new clean-docs refresh-h2o-figures run-catalog run-catalog-report \
	train-template dryrun-template

help:
	@echo "Targets:"
	@echo "  venv           Create .venv with Python (prefers python3.12) and install requirements"
	@echo "  install        Alias for venv"
	@echo "  train          Run PyTorch training (CONFIG=path, NOTES=\"what changed\", PULL=true)"
	@echo "  automl-h2o     Run H2O AutoML training (AUTOML_CONFIG=path, NOTES=..., PULL=true)"
	@echo "  cpu-train      Run training on CPU with minimal threads (CONFIG=..., PULL=true)"
	@echo "  select         Run feature selection (CONFIG=..., METHOD=mi|l1)"
	@echo "  dict           Generate column dictionary (CONFIG=..., CSV optional)"
	@echo "  explore        Explore dataset (CONFIG=..., CSV=path optional)"
	@echo "  dryrun         Run PyTorch training as a dry run (no artifacts persisted)"
	@echo "  dryrun-h2o     Run H2O AutoML as a dry run (no artifacts persisted)"
	@echo "  dryrun-h2o-cv  Run H2O AutoML temporal CV smoke test (2 folds; no artifacts)"
	@echo "  dryrun-cv      Run PyTorch temporal CV smoke test (2 folds; no artifacts persisted)"
	@echo "  cv-train       Run PyTorch temporal CV then full training (smoke config)"
	@echo "  train-template Train the template (sklearn) backend from config (CONFIG=...)"
	@echo "  dryrun-template Run template backend as a dry run (no artifacts)"
	@echo "  run-catalog    Index local_runs and emit _catalog.json (RUNS_ROOT=local_runs)"
	@echo "  wandb-login    Login to W&B using env (WANDB_API_KEY, WANDB_ENTITY)"
	@echo "  pull-run       Download a W&B run into ./wandb-history/<run_id> (RUN=entity/project/run_id | project/run_id | run_id)"
	@echo "  pull-all       Download all W&B runs into ./wandb-history/<run_id> (ENTITY/PROJECT from env/config)"
	@echo "  clean-cloud-history Delete all W&B runs (and logged artifacts) for project (ENTITY/PROJECT from env/config; FORCE=1)"
	@echo "  clean-local-runs Remove local run folders (local_runs/)"
	@echo "  clean-selection-runs Remove feature selection runs (selection_runs/)"
	@echo "  clean-wandb-local Remove local W&B folder (./wandb)"
	@echo "  clean-local-history Remove W&B history folder (./wandb-history)"
	@echo "  clean-all-local  Remove local_runs/, selection_runs/, ./wandb, and ./wandb-history"
	@echo "  clean-venv     Remove the .venv folder"
	@echo "  deps-tools     Install pip-tools into the venv"
	@echo "  deps-compile   Compile requirements.in -> requirements.txt (pinned)"
	@echo "  deps-sync      Sync venv to requirements.txt (exact)"
	@echo "  marker-install Install optional marker-pdf tooling"
	@echo "  marker-pdf     Convert a PDF to Markdown (MARKER_PAPER=... [MARKER_PAGE_RANGE=... MARKER_OUTDIR=...])"
	@echo "  docs           Build platform docs (compiled spec)"
	@echo "  docs-canon     Build docs/architecture/PLATFORM_SPEC.md from ADRs + Journal"
	@echo "  docs-journal-new  Create a new Journal entry (TITLE=..., [TAGS=...], [ADRS=...])"
	@echo "  clean-docs     Remove generated docs (compiled spec)"

$(VENV)/bin/activate: requirements.txt
	@echo "Using Python: $(PYTHON_BIN)"
	$(PYTHON_BIN) -m venv $(VENV)
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

venv: $(VENV)/bin/activate

install: venv

# Usage: make train CONFIG=configs/pytorch_default.yaml (PyTorch backend only; use automl-h2o for H2O AutoML)
CONFIG ?= configs/pytorch_default.yaml
train: venv
	@mkdir -p local_runs
	@echo "Starting PyTorch training in a background shell."; \
	$(SAFE_ENV) $(PY) -m src.cli.pytorch.train --config $(CONFIG) $(if $(NOTES),--notes "$(NOTES)",) $(if $(PULL),--pull,) & \
	PID=$$!; \
	trap 'echo "Interrupt received; stopping PyTorch training (PID $$PID)..."; kill $$PID 2>/dev/null; wait $$PID; exit 130' INT TERM; \
	echo "PyTorch training started (PID $$PID). Waiting for completion..."; \
	if wait $$PID; then \
	  trap - INT TERM; \
	  echo "PyTorch training finished successfully."; \
	else \
	  STATUS=$$?; \
	  trap - INT TERM; \
	  echo "PyTorch training failed with exit code $$STATUS."; \
	  exit $$STATUS; \
	fi

# Usage: make automl-h2o [AUTOML_CONFIG=configs/h2o_default.yaml]
automl-h2o: venv
	@mkdir -p local_runs
	@echo "Starting H2O AutoML training in a background shell."; \
	H2O_BALANCE_CLASSES=$(H2O_BALANCE) \
	PIPELINE_OVERSAMPLING_ENABLED=$(H2O_OVERSAMPLING) \
	H2O_MAX_AFTER_BALANCE_SIZE=$(H2O_MAX_AFTER_BALANCE) \
	H2O_CLASS_SAMPLING_FACTORS=$(H2O_CLASS_SAMPLING_FACTORS) \
	$(SAFE_ENV) $(PY) -m src.cli.h2o.train --config $(AUTOML_CONFIG) $(if $(NOTES),--notes "$(NOTES)",) $(if $(PULL),--pull,) & \
	PID=$$!; \
	trap 'echo "Interrupt received; stopping H2O AutoML training (PID $$PID)..."; kill $$PID 2>/dev/null; wait $$PID; exit 130' INT TERM; \
	echo "H2O AutoML training started (PID $$PID). Waiting for completion..."; \
	if wait $$PID; then \
	  trap - INT TERM; \
	  echo "H2O AutoML training finished successfully."; \
	else \
	  STATUS=$$?; \
	  trap - INT TERM; \
	  echo "H2O AutoML training failed with exit code $$STATUS."; \
	  exit $$STATUS; \
	fi

# Usage: make refresh-h2o-figures RUN_DIR=local_runs/<...>/run_<timestamp>
RUN_DIR ?=
refresh-h2o-figures: venv
	@if [ -z "$(RUN_DIR)" ]; then echo "Set RUN_DIR=path/to/run_dir (containing h2o_leaderboard.csv)"; exit 1; fi
	$(SAFE_ENV) $(PY) -m src.cli.refresh_h2o_figures --run-dir "$(RUN_DIR)"

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
	$(PY) -m src.cli.pytorch.train --config $(CONFIG) --cpu $(if $(NOTES),--notes "$(NOTES)",) $(if $(PULL),--pull,)

# Usage: make select CONFIG=configs/pytorch_default.yaml METHOD=mi
METHOD ?= mi
select: venv
	$(PY) -m src.cli.select --config $(CONFIG) --method $(METHOD)

# Usage: make dict CONFIG=configs/pytorch_default.yaml CSV=data/raw/samples/first_10k_rows.csv
CSV ?=
dict: venv
	$(PY) -m src.cli.gen_column_dict --config $(CONFIG) $(if $(CSV),--csv $(CSV),)

# Usage: make explore CONFIG=configs/pytorch_default.yaml [CSV=data/raw/full/thesis_data_full.csv]
CSV ?=
explore: venv
	$(SAFE_ENV) $(PY) -m src.cli.explore --config $(CONFIG) $(if $(CSV),--csv $(CSV),)

# Usage: make dryrun CONFIG=configs/pytorch_default.yaml
dryrun: venv
	$(SAFE_ENV) $(PY) -m src.cli.pytorch.dryrun --config $(CONFIG)

# Usage: make dryrun-h2o [AUTOML_CONFIG=configs/h2o_default.yaml]
dryrun-h2o: venv
	$(SAFE_ENV) $(PY) -m src.cli.h2o.dryrun --config $(AUTOML_CONFIG)

# Usage: make dryrun-h2o-cv (uses a tiny CV config; H2O backend)
dryrun-h2o-cv: venv
	$(SAFE_ENV) $(PY) -m src.cli.h2o.dryrun --config configs/h2o/cv_smoke.yaml

# Usage: make dryrun-cv (uses a tiny CV config; PyTorch backend)
dryrun-cv: venv
	$(SAFE_ENV) $(PY) -m src.cli.pytorch.dryrun --config configs/pytorch/cv_smoke.yaml

# Usage: make cv-train (PyTorch backend; temporal CV followed by full-data training)
cv-train: venv
	$(SAFE_ENV) $(PY) -m src.cli.pytorch.train --config configs/pytorch/cv_full_train_smoke.yaml $(if $(NOTES),--notes "$(NOTES)",)

# Usage: make train-template [CONFIG=configs/template_default.yaml]
train-template: venv
	$(SAFE_ENV) $(PY) -m src.cli.template.train --config $(or $(CONFIG),configs/template_default.yaml)

# Usage: make dryrun-template [CONFIG=configs/template_default.yaml]
dryrun-template: venv
	$(SAFE_ENV) $(PY) -m src.cli.template.dryrun --config $(or $(CONFIG),configs/template_default.yaml)

# Usage: make run-catalog [RUNS_ROOT=local_runs] [OUT=path]
RUNS_ROOT ?= local_runs
OUT ?=

run-catalog: venv
	$(SAFE_ENV) $(PY) -m src.cli.run_catalog --runs-root $(RUNS_ROOT) $(if $(OUT),--out $(OUT),)

# Usage: make run-catalog-report [RUNS_ROOT=local_runs] [CATALOG=<runs_root>/_catalog.json] [OUT=<runs_root>/index.md]
CATALOG ?=
run-catalog-report: venv
	$(SAFE_ENV) $(PY) -m src.cli.run_catalog_report --runs-root $(RUNS_ROOT) $(if $(CATALOG),--catalog $(CATALOG),) $(if $(OUT),--out $(OUT),)

# W&B helpers
wandb-login: venv
	$(PY) -m src.cli.wandb_login

# Usage: make pull-run RUN=entity/project/run_id | project/run_id | run_id [CONFIG=...]
RUN ?=
FORCE ?=
pull-run: venv
	@if [ -z "$(RUN)" ]; then echo "Set RUN=entity/project/run_id | project/run_id | run_id (use pull-all to sync a project)"; exit 1; fi
	$(PY) -m src.cli.wandb_pull --run $(RUN) --config $(CONFIG) $(if $(FORCE),--force,)

# Usage: make pull-all [ENTITY=...] [PROJECT=...] [CONFIG=configs/pytorch_default.yaml]
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

clean-selection-runs:
	@if [ -d selection_runs ]; then \
	  find selection_runs -mindepth 1 ! -name '.gitignore' -exec rm -rf {} +; \
	fi

clean-all-local: clean-local-runs clean-selection-runs clean-wandb-local clean-local-history


# Dependency management via pip-tools
deps-tools: venv
	$(PIP) install pip-tools

deps-compile: venv
	$(PIP_COMPILE) requirements.in -o requirements.txt

deps-sync: venv
	$(PIP_SYNC) requirements.txt

MARKER_PAPER ?=
MARKER_PAGE_RANGE ?=
MARKER_OUTDIR ?= docs/thesis/bibliography/papers_md

marker-install: venv
	$(PIP) install -r requirements-marker.txt

marker-pdf: marker-install
	@if [ -z "$(MARKER_PAPER)" ]; then echo "Set MARKER_PAPER=path/to/file.pdf"; exit 1; fi
	$(VENV)/bin/marker_single $(MARKER_PAPER) --output_dir $(MARKER_OUTDIR) $(if $(MARKER_PAGE_RANGE),--page_range $(MARKER_PAGE_RANGE),)


# -----------------------------------------------------------------------------
# Docs helpers — Canon and Journal (Makefile-first)
# -----------------------------------------------------------------------------

JOURNAL_DIR ?= docs/architecture/journal
CANON_OUT ?= docs/architecture/PLATFORM_SPEC.md

docs: docs-canon


docs-canon:
	@echo "Building $(CANON_OUT) from ADRs + Journal..."
	@mkdir -p docs/architecture
	@echo "# Platform Specification (Generated)" > $(CANON_OUT)
	@echo >> $(CANON_OUT)
	@echo "- Generated: $$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $(CANON_OUT)
	@echo "- Source: docs/architecture/ADRs + $$(basename $(JOURNAL_DIR))" >> $(CANON_OUT)
	@echo >> $(CANON_OUT)
	@echo "## How to update" >> $(CANON_OUT)
	@echo "- Write ADRs under docs/architecture/ADRs (accepted/proposed)." >> $(CANON_OUT)
	@echo "- Add change entries under $(JOURNAL_DIR) (use make docs-journal-new)." >> $(CANON_OUT)
	@echo "- Rebuild with: make docs-canon" >> $(CANON_OUT)
	@echo >> $(CANON_OUT)
	@echo "---" >> $(CANON_OUT)
	@echo >> $(CANON_OUT)
	@echo "## Decisions — Accepted" >> $(CANON_OUT)
	@for f in $$(ls -1 docs/architecture/ADRs/accepted/*.md 2>/dev/null | sort); do \
	  { echo; echo "### $$(basename $$f)"; echo; cat "$$f"; echo; echo "---"; } >> $(CANON_OUT); \
	done
	@echo >> $(CANON_OUT)
	@echo "## Decisions — Proposed" >> $(CANON_OUT)
	@for f in $$(ls -1 docs/architecture/ADRs/proposed/*.md 2>/dev/null | sort); do \
	  { echo; echo "### $$(basename $$f)"; echo; cat "$$f"; echo; echo "---"; } >> $(CANON_OUT); \
	done
	@echo >> $(CANON_OUT)
	@echo "## Change & Decision Journal" >> $(CANON_OUT)
	@for f in $$(ls -1 $(JOURNAL_DIR)/*.md 2>/dev/null | sort -r); do \
	  { echo; echo "### $$(basename $$f)"; echo; cat "$$f"; echo; echo "---"; } >> $(CANON_OUT); \
	done
	@echo "Wrote $(CANON_OUT)"

clean-docs:
	rm -f $(CANON_OUT)

TITLE ?=
TAGS ?=
ADRS ?=

docs-journal-new:
	@if [ -z "$(TITLE)" ]; then echo "Set TITLE=short-title (e.g., TITLE=Adopt temporal CV for selection)"; exit 1; fi
	@mkdir -p $(JOURNAL_DIR)
	@slug=$$(echo "$(TITLE)" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$$//g'); \
	file="$(JOURNAL_DIR)/$$(date +%Y-%m-%d)-$${slug}.md"; \
	{
	  echo "# $(TITLE)"; \
	  echo; \
	  echo "- Date: $$(date +%Y-%m-%d)"; \
	  echo "- Status: planned"; \
	  if [ -n "$(TAGS)" ]; then echo "- Tags: $(TAGS)"; fi; \
	  echo; \
	  echo "## Summary"; \
	  echo "<one to three sentences about what changed and why>."; \
	  echo; \
	  if [ -n "$(ADRS)" ]; then \
	    echo "## ADRs"; \
	    IFS=','; for a in $(ADRS); do \
	      an=$$(echo $$a | sed -E 's/^0*([0-9]+)/\1/'); \
	      printf -- "- %s — see docs/ADRs (search for %s)\n" "$$a" "$$a"; \
	    done; \
	    echo; \
	  fi; \
	  echo "## Impact"; \
	  echo "- configs: (list keys)"; \
	  echo "- Make: (list targets)"; \
	  echo "- code: (paths e.g., src/... )"; \
	  echo; \
	  echo "## Next"; \
	  echo "- (optional follow-ups)"; \
	} > "$$file"; \
	printf "Created %s\n" "$$file"
