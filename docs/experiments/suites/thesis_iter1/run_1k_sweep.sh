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

usage() {
  echo "Usage: $(basename "$0") [-n \"notes text\"] [--pull]" >&2
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
  "docs/experiments/suites/thesis_iter1/h2o/1k/agnostic.yaml"
  "docs/experiments/suites/thesis_iter1/h2o/1k/selected.yaml"
  "docs/experiments/suites/thesis_iter1/h2o/1k/aware.yaml"
  "docs/experiments/suites/thesis_iter1/h2o/1k/selected_plus_providers.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "--- Running $cfg"
  if [[ -n "$NOTES_ARG" ]]; then
    make automl-h2o AUTOML_CONFIG="$cfg" NOTES="$NOTES_ARG" ${PULL_ARG:+PULL=1}
  else
    make automl-h2o AUTOML_CONFIG="$cfg" ${PULL_ARG:+PULL=1}
  fi
done

echo "Sweep finished. Artifacts under ./local_runs and on W&B if enabled."

