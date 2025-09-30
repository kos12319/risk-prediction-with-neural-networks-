#!/usr/bin/env bash
set -euo pipefail

# Run the streamlined H2O sweep for the LendingClub full-feature experiment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi
cd "$REPO_ROOT"

NOTES_ARG=""
PULL_ARG=""
RESUME=0

usage() {
  echo "Usage: $(basename "$0") [-n \"notes text\"] [--pull] [--resume]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--notes)
      shift || { usage; exit 1; }
      NOTES_ARG="$1"
      ;;
    --pull)
      PULL_ARG=1
      ;;
    --resume)
      RESUME=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift || true
done

CONFIGS=(
  "docs/experiments/suites/lc_full_feature_sweep/h2o/provider_agnostic_all.yaml"
  "docs/experiments/suites/lc_full_feature_sweep/h2o/provider_agnostic_l1.yaml"
  "docs/experiments/suites/lc_full_feature_sweep/h2o/provider_aware_l1.yaml"
  "docs/experiments/suites/lc_full_feature_sweep/h2o/provider_aware_l1_cv.yaml"
)

LOG_PATH="docs/experiments/suites/lc_full_feature_sweep/h2o/run_h2o_suite.log"
mkdir -p "$(dirname "$LOG_PATH")"

RG_AVAILABLE=0
if command -v rg >/dev/null 2>&1; then
  RG_AVAILABLE=1
elif [[ "$RESUME" -eq 1 ]]; then
  echo "Warning: ripgrep (rg) not found; falling back to grep for resume scanning." >&2
fi

DONE_CFGS=""
if [[ "$RESUME" -eq 1 && -f "$LOG_PATH" ]]; then
  if [[ "$RG_AVAILABLE" -eq 1 ]]; then
    while IFS= read -r line; do
      cfg_path="${line#OK: }"
      DONE_CFGS="$DONE_CFGS
$cfg_path"
    done < <(rg -h '^OK: ' "$LOG_PATH" || true)
  else
    while IFS= read -r line; do
      cfg_path="${line#OK: }"
      DONE_CFGS="$DONE_CFGS
$cfg_path"
    done < <(grep -h '^OK: ' "$LOG_PATH" || true)
  fi
fi

for cfg in "${CONFIGS[@]}"; do
  if printf '%s\n' "$DONE_CFGS" | grep -Fxq "$cfg"; then
    echo "Skipping already completed config: $cfg"
    continue
  fi

  echo "--- $(date -Iseconds) :: running $cfg"
  if [[ -n "$NOTES_ARG" ]]; then
    if make automl-h2o AUTOML_CONFIG="$cfg" NOTES="$NOTES_ARG" ${PULL_ARG:+PULL=1}; then
      echo "OK: $cfg" | tee -a "$LOG_PATH"
    else
      status=$?
      echo "FAILED: $cfg (exit $status)" | tee -a "$LOG_PATH"
    fi
  else
    if make automl-h2o AUTOML_CONFIG="$cfg" ${PULL_ARG:+PULL=1}; then
      echo "OK: $cfg" | tee -a "$LOG_PATH"
    else
      status=$?
      echo "FAILED: $cfg (exit $status)" | tee -a "$LOG_PATH"
    fi
  fi
done

echo "H2O suite complete."
