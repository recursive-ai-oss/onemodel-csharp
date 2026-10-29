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

# Optional: wait until the machine is quiet (fewer than BENCH_QUIET_MAX busy cores, sampled over 5 s) before timing, up to
# BENCH_QUIET_WAIT seconds; the wait and the last sample are logged.
if [ -n "${BENCH_QUIET_WAIT:-}" ] && [ -r /proc/stat ]; then
  waited=0
  while :; do
    read -r -a before < /proc/stat; sleep 5; read -r -a after < /proc/stat
    # /proc/stat fields after "cpu": user nice system idle iowait irq softirq steal
    busy=0; idle=0
    for f in 1 2 3 6 7 8; do busy=$(( busy + after[f] - before[f] )); done
    for f in 4 5; do idle=$(( idle + after[f] - before[f] )); done
    nproc_=$(nproc); cores=$(awk -v b=$busy -v i=$idle -v n=$nproc_ 'BEGIN{printf "%.2f", (b+i)>0 ? n*b/(b+i) : 0}')
    echo "$(date -u +%T) busy cores $cores (waited ${waited}s)" >> "$out/quiet.log"
    if awk -v c=$cores -v m="${BENCH_QUIET_MAX:-1.5}" 'BEGIN{exit !(c<m)}'; then break; fi
    waited=$((waited+5))
    if [ "$waited" -ge "$BENCH_QUIET_WAIT" ]; then echo "machine not quiet after ${waited}s" >> "$out/quiet.log"; break; fi
  done
fi

( while true; do echo "$(date -u +%T) $(cat /proc/loadavg 2>/dev/null || uptime)" >> "$out/load.log"; sleep 10; done ) &
sampler=$!
trap 'kill $sampler 2>/dev/null; [ -n "${BENCH_LOCK:-}" ] && rm -f "$BENCH_LOCK"' EXIT

dotnet "$ONEMODEL_ARTIFACTS/bin/Micro/release/RecursiveAI.OneModel.Micro.dll" attention-spike "$reps" "$out/attention-spike.md"
