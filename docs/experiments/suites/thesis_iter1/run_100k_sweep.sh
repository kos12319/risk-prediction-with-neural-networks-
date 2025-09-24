#!/usr/bin/env bash
set -euo pipefail

# Run the 100k H2O sweep (agnostic, selected, aware, selected_plus_providers)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi
cd "$REPO_ROOT"

NOTES_ARG=""
PULL_ARG=""

usage() {
  echo "Usage: $(basename "$0") [-n \"notes text\"] [--pull]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--notes) shift; NOTES_ARG="$1" ;;
    --pull) PULL_ARG=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
  shift || true
done

declare -a CONFIGS=(
  "docs/experiments/suites/thesis_iter1/h2o/100k/agnostic.yaml"
  "docs/experiments/suites/thesis_iter1/h2o/100k/selected.yaml"
  "docs/experiments/suites/thesis_iter1/h2o/100k/aware.yaml"
  "docs/experiments/suites/thesis_iter1/h2o/100k/selected_plus_providers.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "--- Running $cfg"
  if [[ -n "$NOTES_ARG" ]]; then
    make automl-h2o AUTOML_CONFIG="$cfg" NOTES="$NOTES_ARG" ${PULL_ARG:+PULL=1}
  else
    make automl-h2o AUTOML_CONFIG="$cfg" ${PULL_ARG:+PULL=1}
  fi
done

echo "100k sweep finished."

