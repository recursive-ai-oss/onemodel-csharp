#!/usr/bin/env bash
# Builds bench/Micro in Release and runs the attention spike (fast versus exact attention, softmax and transcendentals),
# writing the tables and a load log into <outdir>.
#
#   bench/Micro/attention-spike.sh <outdir> [reps]
#
# Needs `dotnet` (.NET 10 SDK) on PATH. When BENCH_LOCK is set to a path, the script claims that file atomically before timing
# (waiting up to BENCH_LOCK_WAIT seconds, default 3600) and removes it on exit; BENCH_LABEL is written into it.
# ONEMODEL_SPIKE_THREADS, ONEMODEL_SPIKE_SHAPES and ONEMODEL_SPIKE_PARTS are passed through (see AttentionSpike.cs).
set -u
out=${1:?usage: attention-spike.sh <outdir> [reps]}
reps=${2:-3}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
out=$(cd "$out" && pwd)

if [ -n "${BENCH_LOCK:-}" ]; then
  waited=0
  while ! ( set -o noclobber; echo "${BENCH_LABEL:-attention-spike $(date -u +%FT%TZ)}" > "$BENCH_LOCK" ) 2>/dev/null; do
    if [ "$waited" -ge "${BENCH_LOCK_WAIT:-3600}" ]; then echo "lock still held after ${waited}s: $(cat "$BENCH_LOCK")" >&2; exit 3; fi
    sleep 15; waited=$((waited + 15))
  done
  trap 'rm -f "$BENCH_LOCK"' EXIT
fi

export ONEMODEL_ARTIFACTS="${ONEMODEL_ARTIFACTS:-$out/artifacts}"
dotnet build "$here/Micro.csproj" -c Release -v quiet -clp:NoSummary || exit 1

( while true; do echo "$(date -u +%T) $(cat /proc/loadavg 2>/dev/null || uptime)" >> "$out/load.log"; sleep 10; done ) &
sampler=$!
trap 'kill $sampler 2>/dev/null; [ -n "${BENCH_LOCK:-}" ] && rm -f "$BENCH_LOCK"' EXIT

dotnet "$ONEMODEL_ARTIFACTS/bin/Micro/release/RecursiveAI.OneModel.Micro.dll" attention-spike "$reps" "$out/attention-spike.md"
