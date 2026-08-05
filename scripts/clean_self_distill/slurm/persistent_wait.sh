#!/usr/bin/env bash
# Signal-safe wait helpers for the persistent training Slurm wrapper.

CSD_WAIT_INTERRUPTED=0

csd_forward_usr1() {
  CSD_WAIT_INTERRUPTED=1
  if [[ -n "${CSD_CHILD_PID:-}" ]]; then
    kill -USR1 "$CSD_CHILD_PID" 2>/dev/null || true
  fi
}

csd_wait_for_child() {
  local child_pid=$1
  local child_rc

  while true; do
    CSD_WAIT_INTERRUPTED=0
    if wait "$child_pid"; then
      child_rc=0
    else
      child_rc=$?
    fi

    # Bash returns 128 + signal when a trapped signal interrupts wait. The
    # child has not been reaped in that case, so wait for this same PID again.
    if (( child_rc > 128 && CSD_WAIT_INTERRUPTED )); then
      continue
    fi
    return "$child_rc"
  done
}
