#!/usr/bin/env python3
"""Automate repeated Codex CLI runs in non-interactive exec mode.

This script feeds a fixed orchestration prompt to the Codex CLI, relaunching the
process after each completion until either two hours have elapsed or the Codex
command is unavailable. The Codex CLI command can be overridden with the
``CODEX_CLI_CMD`` environment variable (space-separated). Otherwise it defaults
to ``codex exec --full-auto``.

Each launch is logged to ``coding_agent.log`` with best-effort timestamps so the
operator can track activity. The script exits immediately if the Codex binary
cannot be found. To avoid runaway loops when the CLI is misconfigured, the
script backs off between failures and stops after a configurable number of
consecutive errors.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List

PROMPT = """Start addressing the pain points in docs/architecture. Start with the high levle pain points and work down from there. for each pain point you fix run a very very small run to make sure it is done. however do not stop until all are fixed or you are truely stuck. think hard. After you fix each pain point and you make sure it works, review your work, improve it and commit. COMPACT whenever the context left is below 50%. always make runs to test that they work. in the pain points skip changes if you are not 100% sure they are green and have caveats for the nature of the project.

And one more point to consider: decouple the two backends altogether. you may allow them to share some common parts of the code base but i don't like that they have common cli, common configs and common pipeline. The framework in the project should keep common functionality on one hand, but have clear separation of backends so that a third can enter the picture in the future and so that the codebase is cleanly separated. by the way this is high priority for me.

if you manage to finish all the above, start with the architecture future extensions. again run test runs ot make sure everything works.

Mark everything you fix in the relevant point in the docs. After you fix something go over the codebase if you thinkg there might be changes to do elsewhere. update readme's docs agent md gemini md accordingly.

Right before each commit go over the architecture and refine the pain points or the future extentsion. add these changes to commit. use a descriptive message for all your changes, after reviewing them. 

stop your run after each commit. you will be run in a non interctive mode and a script will relaunch you to start again and fix the next item. if all items are fixed, and you cannot refine more, send a signal to kill the running script
if you get stuck, kill the script. DO NOT STAY IN A LOOP FOR MORE THAN 15mins. kill the script so this all stops. Before tou stop your run or kill the script, write a log line with timestamp and the reason you stopped your run or killed the script in coding_agent.log. Make sure you consult this log every time you start to see and understand your status and act according to these instructions."""

DEFAULT_COMMAND = "codex exec --dangerously-bypass-approvals-and-sandbox"
LOG_PATH = Path("coding_agent.log")
MAX_RUNTIME_SEC = 2 * 60 * 60  # two hours
SLEEP_BETWEEN_RUNS = float(os.environ.get("CODEX_CLI_SLEEP", "0"))
FAIL_SLEEP_SEC = float(os.environ.get("CODEX_CLI_FAIL_SLEEP", "5"))
MAX_CONSECUTIVE_FAILURES = int(os.environ.get("CODEX_CLI_MAX_FAILURES", "100"))
RUN_CWD = os.environ.get("CODEX_CLI_CWD")


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} {message}\n")
    except Exception:
        pass


def _resolve_command() -> List[str]:
    raw_cmd = os.environ.get("CODEX_CLI_CMD", DEFAULT_COMMAND)
    if isinstance(raw_cmd, str):
        return shlex.split(raw_cmd)
    return list(raw_cmd)


def main() -> int:
    start_time = time.monotonic()
    command = _resolve_command()

    if not command:
        print("Codex command is empty; aborting.", file=sys.stderr)
        return 2

    _log(f"codex-runner started command={' '.join(command)}")
    iteration = 0
    consecutive_failures = 0

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= MAX_RUNTIME_SEC:
            _log("codex-runner reached two-hour limit; stopping")
            return 0

        iteration += 1
        _log(f"launching iteration={iteration}")

        try:
            completed = subprocess.run(
                command,
                input=PROMPT,
                text=True,
                capture_output=False,
                check=False,
                cwd=RUN_CWD,
            )
        except FileNotFoundError:
            err_msg = f"Codex command not found: {command[0]}"
            print(err_msg, file=sys.stderr)
            _log(err_msg)
            return 1
        except Exception as exc:  # pragma: no cover - defensive logging
            err_msg = f"Codex launch failed: {exc}"
            print(err_msg, file=sys.stderr)
            _log(err_msg)
            return 1

        rc = completed.returncode
        _log(f"iteration={iteration} exited rc={rc}")

        if rc != 0:
            consecutive_failures += 1
            if MAX_CONSECUTIVE_FAILURES and consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                _log(
                    "max consecutive failures reached; stopping runner"
                )
                return rc or 1
            if FAIL_SLEEP_SEC > 0:
                _log(f"sleeping {FAIL_SLEEP_SEC:.1f}s after failure")
                time.sleep(FAIL_SLEEP_SEC)
            continue

        consecutive_failures = 0

        if time.monotonic() - start_time >= MAX_RUNTIME_SEC:
            _log("codex-runner reached two-hour limit after iteration")
            break

        if SLEEP_BETWEEN_RUNS > 0:
            time.sleep(SLEEP_BETWEEN_RUNS)

    return 0


if __name__ == "__main__":
    sys.exit(main())
