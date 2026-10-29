#!/usr/bin/env bash
# Runs a light (non-timing, few-core) job politely on a shared benchmark host:
# refuses to start while a benchmark lock exists, runs at low priority, and stops its own
# process group if a lock appears while it runs.
# Usage: tools/light.sh <command> [args...]    (lock file: $ONEMODEL_LOCK, default ~/WORKER-BENCH.lock)
set -u
lock="${ONEMODEL_LOCK:-$HOME/WORKER-BENCH.lock}"
if [ -e "$lock" ] && ! grep -q 'gpu-train' "$lock"; then
  echo "benchmark lock present: $(cat "$lock"); not starting" >&2
  exit 75
fi
set -m
nice "$@" &
pid=$!
(
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    if [ -e "$lock" ] && ! grep -q 'gpu-train' "$lock"; then
      echo "benchmark lock appeared; stopping job" >&2
      kill -TERM -- "-$pid" 2>/dev/null
      break
    fi
  done
) &
wd=$!
wait "$pid"; rc=$?
kill "$wd" 2>/dev/null
exit $rc
