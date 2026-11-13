#!/usr/bin/env bash
# The bar harness: our .Gpu time against ONNX Runtime's released GPU execution providers, one row per model and batch.
#
#   bar.sh --platform linux|macos --ref webgpu|coreml[,webgpu] [--only NAME] [--phase gpu|cpu] [--cuda] [--out DIR] [--runs N] [--profile] [--ref-only] ROW...
#   bar.sh --rows                      list the rows
#
# gpu phase (default), per row:
#   1. tools/GpuReport runs the model on the desktop GPU through RecursiveAI.OneModel.Gpu (fast mode), checks it against the CPU's exact
#      mode, and exports the inputs and reference outputs as DIR/cases/ROW/case.bin; DIR/native/ROW.txt;
#   2. tools/bar/ort_run.py runs the same model file and the same inputs through the reference provider (each session variant it knows,
#      the best is the reference time): DIR/ref/ROW.json;
#   3. with --cuda, the CUDA provider on the same case, for information: DIR/ref/cuda-ROW.json;
#   4. with --profile, GpuReport again with per-step timing for the rows that miss: DIR/profile/ROW.txt.
# --ref takes a list (macOS: coreml,webgpu): the first provider writes DIR/ref/ROW.json, each other DIR/ref/NAME-ROW.json, and collect.py picks per row
# the faster of the providers whose configuration makes the row a gate row (else the faster one, as an information row). --only NAME runs just that provider.
# --ref-only keeps our numbers and the case files of an earlier run in DIR and re-runs only the reference side.
# cpu phase: the same models on onemodel's CPU path (8 threads) and on the reference runtime's CPU provider, for information: DIR/cpu/ROW.txt,
# DIR/ref/ortcpu-ROW.json. Run it as its own job.
# The machine's load is sampled every 5 s through the run (DIR/load-PHASE.log, Linux also GPU utilisation) next to the device, driver and
# provider pins (DIR/machine-PHASE.json). tools/bar/collect.py then writes DIR/bar-PLATFORM.jsonl and .md.
# Every number is indicative: nothing here is a publishable bench row. Reference providers are black boxes (tools/bar/setup_ort.sh).
# Environment: ONEMODEL_BAR_ORT (where setup_ort.sh put the venvs), ONEMODEL_ARTIFACTS for build output, the Vulkan variables the
# machine needs (VK_ICD_FILENAMES, GLIBC_TUNABLES): validation layers stay off for timing.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
. "$here/bar_rows.sh"
platform=""; ref=""; only=""; phase=gpu; cuda=0; profile=0; runs=20; ours=1
out="${ONEMODEL_ARTIFACTS:-$HOME/onemodel/artifacts}/bar"
while [ $# -gt 0 ]; do
  case "$1" in
    --platform) platform="$2"; shift 2;; --ref) ref="$2"; shift 2;; --only) only="$2"; shift 2;; --phase) phase="$2"; shift 2;; --cuda) cuda=1; shift;;
    --profile) profile=1; shift;; --ref-only) ours=0; shift;; --out) out="$2"; shift 2;; --runs) runs="$2"; shift 2;;
    --rows) echo $ROWS | tr ' ' '\n' | grep -v whisper_greedy | sort | tr '\n' ' '; echo; exit 0;;
    -*) echo "unknown option $1" >&2; exit 2;; *) break;;
  esac
done
[ -n "$platform" ] && [ -n "$ref" ] && [ $# -gt 0 ] || { echo "usage: bar.sh --platform P --ref R [--phase gpu|cpu] [--cuda] [--out DIR] ROW...  (--rows lists them)" >&2; exit 2; }
refs="${ref//,/ }"; ref="${refs%% *}"
ortbase="${ONEMODEL_BAR_ORT:-$ONEMODEL_HOME/bar-ort}"
models="$ONEMODEL_CACHE/models"
mkdir -p "$out/cases" "$out/native" "$out/cpu" "$out/ref" "$out/profile"

# Machine, driver and load -----------------------------------------------------------------------------------------------------
sample() { # time, 1-minute load average, GPU utilisation (%, "-" when unknown)
  local l g="-"
  if [ -r /proc/loadavg ]; then l="$(cut -d' ' -f1 /proc/loadavg)"; else l="$(sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1}')"; fi
  command -v nvidia-smi >/dev/null 2>&1 && g="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  echo "$(date +%s) $l ${g:--}"
}
if [ "$platform" = macos ]; then
  gpu="$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model/{print $2; exit}')"; driver="$(sw_vers -productVersion) (system Metal)"; cpu="$(sysctl -n machdep.cpu.brand_string)"; cores="$(sysctl -n hw.ncpu)"
else
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  cpu="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"; cores="$(nproc)"
fi
cat > "$out/machine-$phase.json" <<EOF
{"os": "$(uname -sr)", "cpu": "$cpu", "cores": $cores, "gpu": "$gpu", "driver": "$driver", "phase": "$phase", "ref": "$ref", "runsPerRow": $runs,
 "date": "$(date -u +%FT%TZ)", "loadBefore": "$(sample | cut -d' ' -f2-)"}
EOF
echo "# time load1 gpu% (job start)" >> "$out/load-$phase.log"
( while true; do sample >> "$out/load-$phase.log"; sleep 5; done ) &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT

# Our side and the reference ---------------------------------------------------------------------------------------------------
# --ref-only keeps the earlier build of our side.
[ "$ours" = 0 ] || dotnet publish -c Release "$here/GpuReport/GpuReport.csproj" -o "$out/gpureport" -v q -clp:NoSummary || exit 1
py="$ortbase/$ref/bin/python"; [ -x "$py" ] || { echo "no $py: run tools/bar/setup_ort.sh $ref" >&2; exit 1; }
cudapy="$ortbase/cuda/bin/python"
# Core ML compiles each model into a 200 MB temporary directory it does not always remove: give it a private one and empty it after every row.
ortt="$out/ort-tmp"; mkdir -p "$ortt"

for r in "$@"; do
  [ -n "$(file_of $r)" ] && [ "$r" != whisper_greedy ] || { echo "unknown row $r" >&2; continue; }
  [ -f "$models/$(file_of $r)" ] || { echo "skip $r: $(file_of $r) not fetched" >&2; continue; }
  if [ "$phase" = cpu ]; then
    if [ "$ours" = 1 ]; then
      # shellcheck disable=SC2086
      dotnet "$out/gpureport/GpuReport.dll" "$models/$(file_of $r)" $(args_of $r) --modes fast --runs 3 --cpu-time 8 --export "$out/cases/$r" > "$out/cpu/$r.txt" 2>&1 || echo "cpu $r failed (see $out/cpu/$r.txt)" >&2
    fi
    # The reference runtime's own CPU provider on the same case: when a GPU provider's time is close to it, the GPU provider is mostly CPU.
    [ -f "$out/cases/$r/case.bin" ] && TMPDIR="$ortt" "$py" "$here/bar/ort_run.py" --provider cpu --model "$models/$(file_of $r)" --case "$out/cases/$r/case.bin" --runs 5 --out "$out/ref/ortcpu-$r.json" 2>&1 | sed "s/^/[$r] /"
    continue
  fi
  if [ "$ours" = 1 ]; then
    # shellcheck disable=SC2086
    dotnet "$out/gpureport/GpuReport.dll" "$models/$(file_of $r)" $(args_of $r) --modes fast --runs "$runs" --export "$out/cases/$r" > "$out/native/$r.txt" 2>&1 || echo "gpu $r failed (see $out/native/$r.txt)" >&2
  fi
  [ -f "$out/cases/$r/case.bin" ] || { echo "skip $r: no case file" >&2; continue; }
  for p in $refs; do
    [ -z "$only" ] || [ "$only" = "$p" ] || continue
    pp="$ortbase/$p/bin/python"; [ -x "$pp" ] || { echo "no $pp: run tools/bar/setup_ort.sh $p" >&2; continue; }
    [ "$p" = "$ref" ] && dst="$out/ref/$r.json" || dst="$out/ref/$p-$r.json"
    TMPDIR="$ortt" "$pp" "$here/bar/ort_run.py" --provider "$p" --model "$models/$(file_of $r)" --case "$out/cases/$r/case.bin" --runs "$runs" --out "$dst" 2>&1 | sed "s/^/[$r] /"
  done
  if [ "$cuda" = 1 ] && [ -x "$cudapy" ]; then
    TMPDIR="$ortt" "$cudapy" "$here/bar/ort_run.py" --provider cuda --model "$models/$(file_of $r)" --case "$out/cases/$r/case.bin" --runs "$runs" --out "$out/ref/cuda-$r.json" 2>&1 | sed "s/^/[$r] /"
  fi
  if [ "$profile" = 1 ]; then
    # shellcheck disable=SC2086
    dotnet "$out/gpureport/GpuReport.dll" "$models/$(file_of $r)" $(args_of $r) --modes fast --runs 5 --profile > "$out/profile/$r.txt" 2>&1 || true
  fi
  rm -rf "${ortt:?}"/* 2>/dev/null
done
sample >> "$out/load-$phase.log"
kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
python3 "$here/bar/collect.py" "$out" --platform "$platform" --ref "${refs// /,}"
