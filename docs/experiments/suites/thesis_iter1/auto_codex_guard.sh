#!/usr/bin/env bash
set -euo pipefail

# Guard script: if the experiments are not running, launch Codex in YOLO (non‑interactive)
# to repair ONLY configs and suite scripts, then resume from where runs failed.
#
# Usage:
#   ./docs/experiments/suites/thesis_iter1/auto_codex_guard.sh [--daemon] [-n "notes"]
#
# Notes:
# - Requires Codex CLI installed (`npm i -g @openai/codex`) and a `yolo` profile in ~/.codex/config.toml
#   that disables sandbox and approvals (as per your local config).
# - Guard edits must be limited to:
#     - docs/experiments/suites/thesis_iter1/h2o/**/_time.yaml (and sibling YAMLs in this suite)
#     - docs/experiments/suites/thesis_iter1/run_*.sh
#     - Optionally: configs/default_automl.yaml (threads/memory/network/logging)
#   (No source code changes are permitted.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi
cd "$REPO_ROOT"

NOTES=${NOTES:-}
DAEMON=0

usage() {
  echo "Usage: $(basename "$0") [--daemon] [-n \"notes\"]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --daemon) DAEMON=1 ;;
    -n|--notes) shift; NOTES="$1" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
  shift || true
done

mkdir -p logs

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a logs/auto_codex_guard.log ; }

experiments_running() {
  ps ax -o pid,command | rg -i "(src\.cli\.automl_h2o|h2o\.jar|run_all_sweeps\.sh)" -n >/dev/null 2>&1
}

experiments_completed() {
  # All sweep logs exist and no FAILED lines remain
  local l1k="logs/run_1k_sweep.log" l10k="logs/run_10k_sweep.log" l100k="logs/run_100k_sweep.log" lfull="logs/run_full_sweep.log"
  if [[ -f "$l1k" && -f "$l10k" && -f "$l100k" && -f "$lfull" ]]; then
    # All sweeps reported finished messages
    rg -q "Sweep finished\.|10k sweep finished\.|100k sweep finished\.|Full sweep finished\." "$l1k" "$l10k" "$l100k" "$lfull" || return 1
    # No failures across any logs
    if ! rg -q "^FAILED: " logs/run_*_sweep.log 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

# Strict completion: all sweeps have their finished marker AND OK count matches expected configs, and no FAILED entries exist
experiments_completed_strict() {
  local scripts=(
    "docs/experiments/suites/thesis_iter1/run_1k_sweep.sh|logs/run_1k_sweep.log|Sweep finished\."
    "docs/experiments/suites/thesis_iter1/run_10k_sweep.sh|logs/run_10k_sweep.log|10k sweep finished\."
    "docs/experiments/suites/thesis_iter1/run_100k_sweep.sh|logs/run_100k_sweep.log|100k sweep finished\."
    "docs/experiments/suites/thesis_iter1/run_full_sweep.sh|logs/run_full_sweep.log|Full sweep finished\."
  )
  for entry in "${scripts[@]}"; do
    IFS='|' read -r script log finish_pat <<<"$entry"
    [[ -f "$log" ]] || return 1
    rg -q "$finish_pat" "$log" || return 1
    mapfile -t expected < <(awk '/declare -a CONFIGS\=\(/, /^\)/ { if ($0 ~ /\.yaml/) { gsub(/["\047,]/, ""); print $1 } }' "$script")
    local expected_count=${#expected[@]}
    local ok_count
    ok_count=$(rg -h '^OK: ' "$log" | wc -l | tr -d ' ')
    if [[ "$ok_count" -lt "$expected_count" ]]; then
      return 1
    fi
  done
  if rg -q '^FAILED: ' logs/run_*_sweep.log 2>/dev/null; then
    return 1
  fi
  return 0
}

collect_failed_cfgs() {
  if ls logs/run_*_sweep.log >/dev/null 2>&1; then
    rg -h "^FAILED: " logs/run_*_sweep.log | sed -E 's/^FAILED: ([^ ]+).*/\1/' | sort -u || true
  fi
}

collect_ok_cfgs() {
  if ls logs/run_*_sweep.log >/dev/null 2>&1; then
    rg -h "^OK: " logs/run_*_sweep.log | sed -E 's/^OK: ([^ ]+).*/\1/' | sort -u || true
  fi
}

launch_codex_repair_and_resume() {
  local failed_list ok_list prompt_file
  failed_list=$(collect_failed_cfgs || true)
  ok_list=$(collect_ok_cfgs || true)
  prompt_file=$(mktemp)

  cat >"$prompt_file" << 'PROMPT'
You are Codex running in non-interactive YOLO mode. Finish within one pass.

Context and constraints:
- Goal: Ensure the H2O AutoML experiments under docs/experiments/suites/thesis_iter1 are running to completion.
- Do NOT modify source code in src/**. You may only adjust:
  - docs/experiments/suites/thesis_iter1/h2o/** (YAML configs)
  - docs/experiments/suites/thesis_iter1/run_*.sh (suite scripts)
  - configs/default_automl.yaml (threads/memory/logging; optional)
- Makefile-first: use `make automl-h2o AUTOML_CONFIG=...` to run configs.
- Respect evaluation invariants (time split, pos_label, thresholding); do not alter protocols.
- If a config fails due to environment or resource issues, prefer adjusting in-config:
  - automl.max_runtime_secs, nthreads, max_mem_size, leaderboard_* settings
  - extends path correctness
  - dataset csv_path
  - avoid editing code; only configs/scripts

What to do:
1) Inspect logs under ./logs/*.log to determine which configs failed and why.
2) Apply minimal changes to the allowed files to fix the failures.
3) Resume by re-running only the configs that failed in this order: 1k, 10k, 100k, full.
   - For each config path, run: `make automl-h2o AUTOML_CONFIG=<cfg> ${PULL:+PULL=1} ${NOTES:+NOTES="$NOTES"}`
4) If no failure list is provided, run the master wrapper: docs/experiments/suites/thesis_iter1/run_all_sweeps.sh

Rerun policy:
- Continue on errors; do not halt the guard.
- Keep per-run budgets: 1k=60s, 10k=300s, 100k=900s, full=5400s.

Output:
- Write a concise summary to stdout: what changed and which runs were relaunched.
PROMPT

  # Append dynamic context
  {
    echo "\nFailed configs to rerun (one per line):";
    if [[ -n "$failed_list" ]]; then echo "$failed_list"; else echo "<none>"; fi
    echo "\nAlready OK configs:";
    if [[ -n "$ok_list" ]]; then echo "$ok_list"; else echo "<none>"; fi
  } >>"$prompt_file"

  # Determine command
  if ! command -v codex >/dev/null 2>&1; then
    log "codex CLI not found in PATH. Install with: npm i -g @openai/codex"
    rm -f "$prompt_file"
    return 1
  fi

  log "Launching Codex in YOLO mode to repair and resume…"
  # Enable apply_patch and plan tools for better non-interactive edits
  # Use working dir as repo root
  codex exec -p yolo -C "$REPO_ROOT" \
    -c include_apply_patch_tool=true \
    -c include_plan_tool=true \
    "$(cat "$prompt_file")" | tee -a logs/auto_codex_guard.log || true

  rm -f "$prompt_file"
}

run_once() {
  if experiments_running; then
    log "Experiments are running; nothing to do."
    return 0
  fi
  if experiments_completed_strict; then
    log "All sweeps completed with no failures; exiting guard."
    exit 0
  fi
  # Attempt targeted resume per sweep first; if failures persist, invoke Codex repair.
  log "Resuming incomplete sweeps (if any)."
  local notes_flags=( )
  [[ -n "$NOTES" ]] && notes_flags=( -n "$NOTES" )
  for tup in \
    "docs/experiments/suites/thesis_iter1/run_1k_sweep.sh|logs/run_1k_sweep.log" \
    "docs/experiments/suites/thesis_iter1/run_10k_sweep.sh|logs/run_10k_sweep.log" \
    "docs/experiments/suites/thesis_iter1/run_100k_sweep.sh|logs/run_100k_sweep.log" \
    "docs/experiments/suites/thesis_iter1/run_full_sweep.sh|logs/run_full_sweep.log"; do
    IFS='|' read -r script log <<<"$tup"
    if [[ -f "$log" ]]; then
      mapfile -t expected < <(awk '/declare -a CONFIGS\=\(/, /^\)/ { if ($0 ~ /\.yaml/) { gsub(/["\047,]/, ""); print $1 } }' "$script" 2>/dev/null || true)
      local expected_count=${#expected[@]:-0}
      local ok_count
      ok_count=$(rg -h '^OK: ' "$log" 2>/dev/null | wc -l | tr -d ' ')
      if [[ "$ok_count" -lt "$expected_count" ]]; then
        log "Resuming $(basename "$script") from remaining configs…"
        bash "$script" --resume "${notes_flags[@]}" --pull || true
      fi
    else
      log "No log for $(basename "$script"); starting fresh."
      bash "$script" "${notes_flags[@]}" --pull || true
    fi
  done

  if ! experiments_completed_strict; then
    log "Post-resume failures detected; invoking Codex repair/resume."
    launch_codex_repair_and_resume || true
  fi
}

if [[ "$DAEMON" -eq 1 ]]; then
  log "Starting guard in daemon mode (check every 5 minutes)."
  while true; do
    if experiments_completed_strict; then
      log "All sweeps completed with no failures; stopping daemon loop."
      exit 0
    fi
    run_once || true
    sleep 300
  done
else
  run_once
fi
