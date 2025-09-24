#!/usr/bin/env bash
set -euo pipefail

# Batch runner for the lc_full_feature_sweep suite
# Runs PyTorch and H2O configs via Makefile targets and writes per-config logs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="${ROOT_DIR}"
LOG_DIR="${SUITE_DIR}/_logs"
mkdir -p "${LOG_DIR}"

timestamp() { date '+%Y-%m-%d_%H-%M-%S'; }

run_pytorch() {
  local cfg="$1"
  local name
  name="pytorch_$(basename "${cfg}" .yaml)_$(timestamp)"
  echo "[PyTorch] Running ${cfg} …"
  MAKEFLAGS= make train CONFIG="${cfg}" NOTES="suite:lc_full_feature_sweep cfg:$(basename "${cfg}")" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

run_h2o() {
  local cfg="$1"
  local name
  name="h2o_$(basename "${cfg}" .yaml)_$(timestamp)"
  echo "[H2O] Running ${cfg} …"
  MAKEFLAGS= make automl-h2o AUTOML_CONFIG="${cfg}" NOTES="suite:lc_full_feature_sweep cfg:$(basename "${cfg}")" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

main() {
  echo "Suite root: ${SUITE_DIR}"
  echo "Logs: ${LOG_DIR}"

  # Fail fast if full dataset is missing
  if [ ! -f "data/raw/full/thesis_data_full.csv" ]; then
    echo "ERROR: data/raw/full/thesis_data_full.csv not found. Please place the full dataset and retry." >&2
    exit 1
  fi

  # PyTorch configs (8 total)
  for cfg in \
    "${SUITE_DIR}/pytorch/agnostic_time.yaml" \
    "${SUITE_DIR}/pytorch/agnostic_time_cv5.yaml" \
    "${SUITE_DIR}/pytorch/agnostic_random.yaml" \
    "${SUITE_DIR}/pytorch/aware_time.yaml" \
    "${SUITE_DIR}/pytorch/aware_time_cv5.yaml" \
    "${SUITE_DIR}/pytorch/aware_random.yaml" \
    "${SUITE_DIR}/pytorch/l1_subset_time.yaml" \
    "${SUITE_DIR}/pytorch/mi_subset_time.yaml" \
  ; do
    run_pytorch "${cfg}"
  done

  # H2O configs (8 total)
  for cfg in \
    "${SUITE_DIR}/h2o/agnostic_time.yaml" \
    "${SUITE_DIR}/h2o/agnostic_random.yaml" \
    "${SUITE_DIR}/h2o/aware_time.yaml" \
    "${SUITE_DIR}/h2o/aware_random.yaml" \
    "${SUITE_DIR}/h2o/mi_subset_time.yaml" \
    "${SUITE_DIR}/h2o/mi_subset_random.yaml" \
    "${SUITE_DIR}/h2o/l1_subset_time.yaml" \
    "${SUITE_DIR}/h2o/l1_subset_random.yaml" \
  ; do
    run_h2o "${cfg}"
  done

  echo "All runs submitted. Check local_runs/ and ${LOG_DIR} for outputs."
}

main "$@"
