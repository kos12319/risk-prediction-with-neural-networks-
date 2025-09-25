#!/usr/bin/env bash
set -euo pipefail

# Run all 4 sweeps (1k, 10k, 100k, full) sequentially.
# Each sub-script continues on per-config failure; this wrapper also continues
# on sub-script failure and timestamps progress to logs.

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

mkdir -p logs

echo "[$(date -Is)] Ensuring venv"
make venv >> logs/run_all_venv.log 2>&1 || echo "venv setup failed (continuing)" >&2

declare -a SCRIPTS=(
  "docs/experiments/suites/thesis_iter1/run_1k_sweep.sh"
  "docs/experiments/suites/thesis_iter1/run_10k_sweep.sh"
  "docs/experiments/suites/thesis_iter1/run_100k_sweep.sh"
  "docs/experiments/suites/thesis_iter1/run_full_sweep.sh"
)

for s in "${SCRIPTS[@]}"; do
  base="$(basename "$s" .sh)"
  log="logs/${base}.log"
  echo "[$(date -Is)] Starting $s | log -> $log"
  if [[ -n "$NOTES_ARG" ]]; then
    if "$s" --pull ${PULL_ARG:+--pull} -n "$NOTES_ARG" >> "$log" 2>&1; then
      echo "[$(date -Is)] OK: $s"
    else
      echo "[$(date -Is)] FAILED: $s (continuing)" >&2
    fi
  else
    if "$s" ${PULL_ARG:+--pull} >> "$log" 2>&1; then
      echo "[$(date -Is)] OK: $s"
    else
      echo "[$(date -Is)] FAILED: $s (continuing)" >&2
    fi
  fi
done

echo "[$(date -Is)] All sweeps submitted. Check ./logs/*.log for details."

