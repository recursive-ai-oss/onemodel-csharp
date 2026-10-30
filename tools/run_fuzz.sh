#!/usr/bin/env bash
# Runs one libFuzzer target against a reader for a fixed time, in a single process, with the Checked build.
#
#   run_fuzz.sh <workdir> <target: onnx|tensor|format|checkpoint> <seconds> <seed-dir>...
#
# <workdir> must already hold what setup_fuzz.sh prepared. Run from the repository root with `dotnet` on PATH. The run stops
# early if ~/WORKER-BENCH.lock appears without "gpu-train" in it, because a timing run is under way. Findings (crash-*,
# oom-*, timeout-*) are left in <workdir>/<target>/findings.
set -u
work=$1; target=$2; seconds=$3; shift 3
repo=$(pwd)
out="$work/$target"
rm -rf "$out"; mkdir -p "$out/corpus" "$out/findings"

dotnet publish fuzz/OneModel.Fuzz -c Checked -o "$out/pub" -v quiet -clp:NoSummary >/dev/null || exit 1
if [ "$target" = checkpoint ]; then
  # The seed is written before instrumentation: instrumented code only runs under the fuzzer.
  dotnet "$out/pub/RecursiveAI.OneModel.Fuzz.dll" seed-checkpoint "$out/seeds" || exit 1
  DOTNET_ROLL_FORWARD=Major "$work/tools/sharpfuzz" "$out/pub/RecursiveAI.OneModel.Training.dll" >/dev/null || exit 1
fi
DOTNET_ROLL_FORWARD=Major "$work/tools/sharpfuzz" "$out/pub/RecursiveAI.OneModel.dll" >/dev/null || exit 1

i=0
for d in "$@"; do
  while IFS= read -r f; do
    i=$((i+1)); cp "$f" "$out/corpus/seed-$i"
  done < <(find "$d" -type f \( -name '*.onnx' -o -name '*.pb' -o -name '*.onemodel' -o -name '*.ckpt' \) -size -256k)
done
if [ -d "$out/seeds" ]; then
  for f in "$out"/seeds/*; do i=$((i+1)); cp "$f" "$out/corpus/gen-$i"; done
fi
echo "seeds: $i"

export ONEMODEL_FUZZ_EXTDIR="$repo/testdata/onnx/external"
setsid "$work/libfuzzer-dotnet" --target_path="$out/pub/RecursiveAI.OneModel.Fuzz" --target_arg="$target" \
  "$out/corpus" -max_total_time="$seconds" -timeout=20 -rss_limit_mb=4096 -artifact_prefix="$out/findings/" -print_final_stats=1 \
  > "$out/fuzz.log" 2>&1 &
pid=$!
trap 'kill -- -$pid 2>/dev/null' EXIT
while kill -0 $pid 2>/dev/null; do
  sleep 20
  if [ -e ~/WORKER-BENCH.lock ] && ! grep -q 'gpu-train' ~/WORKER-BENCH.lock; then
    echo "a timing lock appeared; stopping"; kill -- -$pid; break
  fi
done
wait $pid
echo "exit: $?"
tail -25 "$out/fuzz.log"
ls "$out/findings"
