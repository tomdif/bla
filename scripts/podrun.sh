#!/usr/bin/env bash
# podrun.sh — robust long-run launcher, encoding the process lessons from the
# substrate arc so they can't recur:
#   1. nohup + </dev/null + &        -> survives ssh disconnect (no exit-255 kill)
#   2. python3 -u, redirect to LOG   -> unbuffered, peekable pod-side log
#   3. 60s no-progress health check  -> a wedge-on-init dies in 60s, not 27 min
#
# Usage:  bash scripts/podrun.sh <logfile> <progress-grep-regex> -- <command...>
# Example:
#   bash scripts/podrun.sh train.log '\[step ' -- python3 -u system1_motion/train.py --steps 30000
#
# The health check kills the job if the log shows no line matching the progress
# regex within HEALTH_SECS (default 60). Set HEALTH_SECS=0 to disable.
set -uo pipefail
LOG="$1"; PROGRESS="$2"; shift 2
[ "$1" = "--" ] && shift
HEALTH_SECS="${HEALTH_SECS:-60}"

: > "$LOG"
nohup "$@" > "$LOG" 2>&1 </dev/null &
PID=$!
echo "podrun: PID $PID -> $LOG (progress=/$PROGRESS/, health=${HEALTH_SECS}s)"

if [ "$HEALTH_SECS" -gt 0 ]; then
  ( sleep "$HEALTH_SECS"
    if kill -0 "$PID" 2>/dev/null && ! grep -qE "$PROGRESS" "$LOG" 2>/dev/null; then
      echo "podrun: HEALTH FAIL — no progress in ${HEALTH_SECS}s, killing $PID" | tee -a "$LOG"
      kill -9 "$PID" 2>/dev/null
    fi
  ) &
fi
echo "$PID"
