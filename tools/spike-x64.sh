#!/usr/bin/env bash
# Times the exact and fast matrix-product and convolution kernels on the reference shapes, and prints the machine description
# and the load before, during and after, so the numbers can be read later.
#
#   tools/spike-x64.sh [table|hashes] [output-dir]
#
# Needs a .NET 10 SDK on PATH. Environment:
#   ONEMODEL_ARTIFACTS  where build output goes (default: $HOME/onemodel-artifacts)
#   BENCH_LOCK          path of a lock file to claim for the timing run, atomically; fails if it exists. Unset means no lock.
#   GHZ                 nominal clock in GHz for the "% of peak" columns (default: the CPU's maximum from lscpu)
#   THREADS             the "all cores" thread count (default: logical processors)
# "table" takes about three minutes of timing (every shape, fast and exact, each ISA the CPU has, 1 thread and all threads, three
# repetitions each; the median is reported). "hashes" is a correctness run, not a timing run, and uses at most 8 threads.
set -euo pipefail

mode="${1:-table}"
out="${2:-$PWD/spike-out}"
mkdir -p "$out"
here="$(cd "$(dirname "$0")/.." && pwd)"
export ONEMODEL_ARTIFACTS="${ONEMODEL_ARTIFACTS:-$HOME/onemodel-artifacts}"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1

dotnet build -c Release "$here/bench/Micro" -v quiet -clp:NoSummary
micro="$ONEMODEL_ARTIFACTS/bin/Micro/release/RecursiveAI.OneModel.Micro.dll"

if [ "$mode" = hashes ]; then
  # Each line of the environment list is one run: default, AVX-512 off, AVX2 (and so AVX-512) off, all hardware intrinsics off.
  i=0
  for e in "A=1" "DOTNET_EnableAVX512=0" "DOTNET_EnableAVX512=0 DOTNET_EnableAVX2=0" "DOTNET_EnableHWIntrinsic=0"; do
    i=$((i + 1))
    env $e DOTNET_PROCESSOR_COUNT=8 nice dotnet "$micro" spike hashes > "$out/hashes-$i.txt"
    echo "run $i ($e): $(grep -c . "$out/hashes-$i.txt") lines; $(grep '^# all\|MISMATCH' "$out/hashes-$i.txt" | head -1)"
  done
  exit 0
fi

if [ -n "${BENCH_LOCK:-}" ]; then
  set -o noclobber
  echo "onemodel kernel spike $(date -u +%FT%TZ) expected $(date -u -d '+8 minutes' +%FT%TZ)" > "$BENCH_LOCK" || { echo "lock $BENCH_LOCK is held" >&2; exit 3; }
  trap 'rm -f "$BENCH_LOCK"' EXIT
  set +o noclobber
fi

logical="$(nproc)"
cores="$logical"
if command -v lscpu > /dev/null; then
  sockets="$(lscpu | awk -F: '/^Socket\(s\)/ {gsub(/ /,"",$2); print $2}')"
  per="$(lscpu | awk -F: '/^Core\(s\) per socket/ {gsub(/ /,"",$2); print $2}')"
  [ -n "$sockets" ] && [ -n "$per" ] && cores=$((sockets * per))
  ghz="${GHZ:-$(lscpu | awk -F: '/CPU max MHz/ {printf "%.2f", $2 / 1000}')}"
  vendor="$(lscpu | awk -F: '/Vendor ID/ {gsub(/ /,"",$2); print $2}')"
  {
    lscpu | grep -E 'Model name|Vendor|Socket|Core|Thread|MHz'
    echo "flags: $(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|fma|avx512f|avx512_vnni|avx_vnni)$' | tr '\n' ' ')"
  } > "$out/machine.txt"
else
  ghz="${GHZ:-0}"
  vendor=unknown
fi
# Execution ports for 512-bit fused multiply-add: AMD cores run 512-bit operations as two 256-bit halves.
fma512=2
[ "$vendor" = AuthenticAMD ] && fma512=1
{
  echo "dotnet: $(dotnet --version)"
  echo "logical processors $logical, physical cores $cores, nominal GHz $ghz, 512-bit FMA ports assumed $fma512"
  echo "load before: $(cat /proc/loadavg 2> /dev/null || uptime)"
} | tee -a "$out/machine.txt"

( while true; do echo "$(date +%T) $(cat /proc/loadavg 2> /dev/null || uptime)" >> "$out/load-during.txt"; sleep 10; done ) &
sampler=$!
trap 'kill $sampler 2> /dev/null; rm -f "${BENCH_LOCK:-/nonexistent}"' EXIT

dotnet "$micro" spike table --ghz "$ghz" --fma256 2 --fma512 "$fma512" --cores "$cores" --threads "${THREADS:-$logical}" \
  --csv "$out/table.csv" > "$out/table.md"
echo "load after: $(cat /proc/loadavg 2> /dev/null || uptime)" | tee -a "$out/machine.txt"
echo "wrote $out/table.md, $out/table.csv, $out/machine.txt, $out/load-during.txt"
