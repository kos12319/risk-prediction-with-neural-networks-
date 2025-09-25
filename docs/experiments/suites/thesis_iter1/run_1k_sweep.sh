#!/usr/bin/env bash
set -euo pipefail

# Run the 1k H2O sweep (agnostic, selected, aware, selected_plus_providers)
# using the Makefile target. Designed to be invoked from the repo root, but
# it will also resolve the root relative to this script path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  # Fallback: ascend to repo root from docs/experiments/suites/thesis_iter1
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi

cd "$REPO_ROOT"

NOTES_ARG=""
PULL_ARG=""

RESUME=0
usage() {
  echo "Usage: $(basename "$0") [-n \"notes text\"] [--pull] [--resume]" >&2
  echo "Runs all four 1k configs via 'make automl-h2o'." >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--notes)
      shift
      NOTES_ARG="$1"
      ;;
    --pull)
      PULL_ARG=1
      ;;
    --resume)
      RESUME=1
      ;;
    -h|--help)
      usage; exit 0;
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 1;
      ;;
  esac
  shift || true
done

declare -a CONFIGS=(
  "docs/experiments/suites/thesis_iter1/h2o/1k/agnostic_time.yaml"
  "docs/experiments/suites/thesis_iter1/h2o/1k/selected_time.yaml"
  "docs/experiments/suites/thesis_iter1/h2o/1k/aware_time.yaml"
  "docs/experiments/suites/thesis_iter1/h2o/1k/selected_plus_providers_time.yaml"
)

# If resuming, skip configs already marked OK in the existing log.
LOG_PATH="logs/run_1k_sweep.log"
if [[ "$RESUME" -eq 1 && -f "$LOG_PATH" ]]; then
  mapfile -t DONE_CFGS < <(rg -h '^OK: ' "$LOG_PATH" | sed -E 's/^OK: (.*)$/\1/' | sort -u)
  if [[ ${#DONE_CFGS[@]} -gt 0 ]]; then
    TMP=( )
    for c in "${CONFIGS[@]}"; do
      skip=0
      for d in "${DONE_CFGS[@]}"; do [[ "$c" == "$d" ]] && skip=1 && break; done
      [[ $skip -eq 0 ]] && TMP+=("$c") || true
    done
    CONFIGS=("${TMP[@]}")
  fi
fi

for cfg in "${CONFIGS[@]}"; do
  echo "--- Running $cfg"
  if [[ -n "$NOTES_ARG" ]]; then
    if make automl-h2o AUTOML_CONFIG="$cfg" NOTES="$NOTES_ARG" ${PULL_ARG:+PULL=1}; then
      echo "OK: $cfg"
    else
      echo "FAILED: $cfg (continuing)" >&2
    fi
  else
    if make automl-h2o AUTOML_CONFIG="$cfg" ${PULL_ARG:+PULL=1}; then
      echo "OK: $cfg"
    else
      echo "FAILED: $cfg (continuing)" >&2
    fi
  fi
done

echo "Sweep finished. Artifacts under ./local_runs and on W&B if enabled."
