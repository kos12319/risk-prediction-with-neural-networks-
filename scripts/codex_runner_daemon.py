#!/usr/bin/env python3
"""Daemon supervising the Codex YOLO runner.

The daemon spawns ``run_codex_yolo.py`` and checks every five minutes that it is
still running and has not exceeded the two-hour execution budget. If the child
process stops, the daemon exits without restarting it. If the two-hour limit is
breached, the daemon terminates and kills the child, then stops without relaunching.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RUNNER_PATH = Path(__file__).resolve().parent / "run_codex_yolo.py"
CHECK_INTERVAL_SEC = 5 * 60
MAX_RUNTIME_SEC = 2 * 60 * 60
LOG_PATH = Path("coding_agent.log")


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} daemon {message}\n")
    except Exception:
        pass


def _terminate_process(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        try:
            proc.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


def main() -> int:
    if not RUNNER_PATH.exists():
        print(f"Runner script not found: {RUNNER_PATH}", file=sys.stderr)
        _log(f"runner missing at {RUNNER_PATH}")
        return 1

    cmd = [sys.executable, str(RUNNER_PATH)]
    _log(f"launching runner cmd={' '.join(cmd)}")

    try:
        proc = subprocess.Popen(cmd)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to launch runner: {exc}", file=sys.stderr)
        _log(f"failed to launch runner: {exc}")
        return 1

    start_time = time.monotonic()

    while True:
        time.sleep(CHECK_INTERVAL_SEC)
        elapsed = time.monotonic() - start_time

        poll = proc.poll()
        if poll is not None:
            _log(f"runner exited rc={poll}")
            return 0

        if elapsed >= MAX_RUNTIME_SEC:
            _log("runtime exceeded; terminating runner")
            _terminate_process(proc)
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
