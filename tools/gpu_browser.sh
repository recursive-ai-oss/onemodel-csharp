#!/usr/bin/env bash
# Runs reference-set models on RecursiveAI.OneModel.Gpu in headless Chrome (WebGPU) and next to the desktop GPU path.
#
#   gpu_browser.sh [--out DIR] [--adapters nvidia,swiftshader] [--modes fast,exact] [--runs N] [--no-native] [--ort] [--aot] [--query "&fuse=0"] ROW...
#
# A ROW is a name from the table below. For each row the script
#   1. runs tools/GpuReport natively (Vulkan or Metal): CPU exact outputs and inputs go to DIR/cases/ROW/case.bin, the desktop GPU's
#      parity and wall time to DIR/native/ROW.txt;
#   2. runs the model in Chrome on each adapter and mode (a fresh browser per run, so one model's memory never affects another's):
#      DIR/browser/ADAPTER-MODE-ROW.json;
#   3. with --ort, runs the same case through onnxruntime-web's WebGPU provider (tools/ort_web.sh): DIR/ort/ADAPTER-ROW.json.
# --query appends to the page's query string (fuse=0 runs the model's own operators instead of the fused attention and GELU).
# --aot compiles the browser app ahead of time (needs the wasm-tools workload) and labels its rows ADAPTER+aot; without it the app runs on the
# interpreter.
# Then prints one table (tools/web/summarize.mjs). Model files are the ones the other tools fetch into ONEMODEL_CACHE (tools/export_models.py).
# Environment: NODE (default: node), CHROME_NVIDIA_FLAGS (default below), ONEMODEL_ARTIFACTS for build output, VK_ADD_LAYER_PATH is
# not needed here (the native rows are timing and export rows, not validation rows).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
out="${ONEMODEL_ARTIFACTS:-$HOME/onemodel/artifacts}/gpu-browser"; adapters="nvidia,swiftshader"; modes="fast,exact"; runs=10; native=1; ort=0; aot=0; extra=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out="$2"; shift 2;; --adapters) adapters="$2"; shift 2;; --modes) modes="$2"; shift 2;; --runs) runs="$2"; shift 2;;
    --no-native) native=0; shift;; --ort) ort=1; shift;; --aot) aot=1; shift;; --query) extra="$2"; shift 2;; -*) echo "unknown option $1" >&2; exit 2;; *) break;;
  esac
done
node="${NODE:-node}"
nvidia_flags="${CHROME_NVIDIA_FLAGS:---enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --use-vulkan=native --ignore-gpu-blocklist --disable-vulkan-surface}"
swift_flags="--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist"

. "$here/bar_rows.sh"

[ $# -gt 0 ] || { echo "rows: $ROWS" | tr ' ' '\n' | sort | tr '\n' ' '; echo; exit 2; }
mkdir -p "$out/cases" "$out/native" "$out/browser" "$out/ort"
models="$ONEMODEL_CACHE/models"

if [ "$native" = 1 ]; then
  dotnet publish -c Release "$here/GpuReport/GpuReport.csproj" -o "$out/gpureport" -v q -clp:NoSummary || exit 1
fi
if [ "$aot" = 1 ]; then
  app="$out/app-aot"; tag="+aot"; dotnet publish -c Release "$here/GpuBrowser/GpuBrowser.csproj" -o "$app" -p:RunAOTCompilation=true -v q -clp:NoSummary || exit 1
else
  app="$out/app"; tag=""; dotnet publish -c Release "$here/GpuBrowser/GpuBrowser.csproj" -o "$app" -v q -clp:NoSummary || exit 1
fi
www="$app/wwwroot"

for r in "$@"; do
  [ -n "$(file_of $r)" ] || { echo "unknown row $r" >&2; continue; }
  [ -f "$models/$(file_of $r)" ] || { echo "skip $r: $(file_of $r) not fetched" >&2; continue; }
  if [ "$native" = 1 ]; then
    # shellcheck disable=SC2086
    dotnet "$out/gpureport/GpuReport.dll" "$models/$(file_of $r)" $(args_of $r) --modes fast --runs "$([ "$r" = whisper_greedy ] && echo 1 || echo 20)" --export "$out/cases/$r" > "$out/native/$r.txt" 2>&1 || echo "native $r failed (see $out/native/$r.txt)" >&2
  fi
  [ -f "$out/cases/$r/case.bin" ] || { echo "skip $r: no case file" >&2; continue; }
  for a in ${adapters//,/ }; do
    [ "$a" = nvidia ] && flags="$nvidia_flags" || flags="$swift_flags"
    for m in ${modes//,/ }; do
      # Exact mode is repeatability evidence, not a timing row: two timed runs are enough. SwiftShader is the correctness adapter and runs on
      # the CPU: the first two runs (parity and the repeatability check) are all it needs unless SWIFT_RUNS asks for timed ones.
      n=$runs; w=3; [ "$m" = exact ] && n=2
      [ "$a" = swiftshader ] && { n=${SWIFT_RUNS:-0}; w=${SWIFT_WARMUPS:-0}; }
      q="model=/models/$(file_of $r)&case=/cases/$r/case.bin&mode=$m&runs=$n&warmups=$w&label=$r$extra"
      [ "$r" = whisper_greedy ] && q="$q&greedy=1"
      CHROME_GPU_FLAGS="$flags" "$node" "$here/web/run-page.mjs" --root "$www" --mounts "/models=$models,/cases=$out/cases" \
        --path "/index.html?$q" \
        --timeout 1500 --out "$out/browser/$a$tag-$m-$r.json" --log 1 2> "$out/browser/$a$tag-$m-$r.log" > /dev/null || echo "browser $a $m $r: no result" >&2
    done
    if [ "$ort" = 1 ]; then
      CHROME_GPU_FLAGS="$flags" NODE="$node" "$here/ort_web.sh" --out "$out" --adapter "$a" --runs "$([ "$a" = swiftshader ] && echo 2 || echo "$runs")" "$r"
    fi
  done
done
"$node" "$here/web/summarize.mjs" "$out"
