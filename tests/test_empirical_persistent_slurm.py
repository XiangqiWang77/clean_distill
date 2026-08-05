"""Regression tests for signal-safe persistent Slurm requeue handling."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/clean_self_distill/slurm/persistent_wait.sh"
LAUNCHER = ROOT / "scripts/clean_self_distill/slurm/empirical_persistent.slurm"


def _run_bash(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", body, "persistent-wait-test", str(HELPER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_repeated_usr1_interruptions_keep_waiting_for_the_same_child() -> None:
    result = _run_bash(
        r'''
set -Eeuo pipefail
source "$1"

signal_log=$(mktemp)
ready_file="${signal_log}.ready"
sender_pid=
cleanup() {
  if [[ -n "${sender_pid:-}" ]]; then
    kill "$sender_pid" 2>/dev/null || true
    wait "$sender_pid" 2>/dev/null || true
  fi
  if [[ -n "${CSD_CHILD_PID:-}" ]]; then
    kill "$CSD_CHILD_PID" 2>/dev/null || true
    wait "$CSD_CHILD_PID" 2>/dev/null || true
  fi
  rm -f "$signal_log" "$ready_file"
}
trap cleanup EXIT
trap csd_forward_usr1 USR1

child() {
  local signal_count=0
  trap '
    signal_count=$((signal_count + 1))
    printf "USR1\n" >> "$signal_log"
    if (( signal_count == 2 )); then
      exit 75
    fi
  ' USR1
  : > "$ready_file"
  while true; do
    sleep 0.02
  done
}

child &
CSD_CHILD_PID=$!
child_pid=$CSD_CHILD_PID
while [[ ! -e "$ready_file" ]]; do
  sleep 0.01
done
parent_pid=$$
(
  sleep 0.1
  kill -USR1 "$parent_pid"
  sleep 0.1
  kill -USR1 "$parent_pid"
) &
sender_pid=$!

child_rc=0
csd_wait_for_child "$CSD_CHILD_PID" || child_rc=$?
[[ "$CSD_CHILD_PID" == "$child_pid" ]]
CSD_CHILD_PID=
wait "$sender_pid"
sender_pid=

[[ "$child_rc" == 75 ]]
[[ $(wc -l < "$signal_log") == 2 ]]
'''
    )

    assert result.returncode == 0, result.stderr


def test_wait_returns_true_child_statuses_without_disabling_errexit() -> None:
    result = _run_bash(
        r'''
set -Eeuo pipefail
source "$1"

(exit 0) &
CSD_CHILD_PID=$!
csd_wait_for_child "$CSD_CHILD_PID"

(exit 23) &
CSD_CHILD_PID=$!
child_rc=0
csd_wait_for_child "$CSD_CHILD_PID" || child_rc=$?
[[ "$child_rc" == 23 ]]

# A genuine 128+signal-shaped exit is not an interrupted wait unless the
# wrapper's USR1 trap ran during that wait.
(exit 138) &
CSD_CHILD_PID=$!
child_rc=0
csd_wait_for_child "$CSD_CHILD_PID" || child_rc=$?
CSD_CHILD_PID=
[[ "$child_rc" == 138 ]]
[[ $- == *e* ]]
'''
    )

    assert result.returncode == 0, result.stderr


def test_launcher_uses_signal_safe_wait_before_requeue_decision() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert (
        'source "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/persistent_wait.sh"'
        in launcher
    )
    assert 'csd_wait_for_child "$CSD_CHILD_PID" || CSD_RC=$?' in launcher
    assert launcher.index('csd_wait_for_child "$CSD_CHILD_PID"') < launcher.index(
        "CSD_CHILD_PID=\n\nif (( CSD_RC == 75 ))"
    )
    assert 'scontrol requeue "$SLURM_JOB_ID"' in launcher
    assert "set +e" not in launcher
