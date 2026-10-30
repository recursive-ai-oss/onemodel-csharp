#!/usr/bin/env bash
# One command for the exact-mode golden-hash table on this machine (usable from CI).
#
#   golden.sh [--out DIR] [--models "resnet50:fp32:1 ..."] [--cases DIR] [--every N] [--aot] [--merge FILE ...] [--markdown FILE]
#
# Builds tools/GoldenModels, runs it in exact mode under the rows of tools/golden_hash.py (default threads, one thread, AVX-512 off,
# AVX2 off or the arm64 equivalent, hardware intrinsics off, and NativeAOT with --aot), and prints whether every hash is equal.
# Saves this machine's results to DIR/results-<os>-<arch>.json; --merge adds results saved elsewhere (other machines, browser-wasm).
# Models come from the fetch tools (see tools/env.sh for locations); a model that is not present is left out.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
out="$ONEMODEL_HOME/golden"; cases="$ONEMODEL_CACHE/onnx-backend-test/node"; every=1; aot=0; merge=(); md=""
rows="resnet50:fp32:1 mobilenet_v3_large:fp32:1 bert_base:fp32:1 minilm_l6:fp32:1 whisper_small:encoder_fp32:1 whisper_small:decoder_fp32:1 crnn:fp32:1-val-00000539 dbnet:fp32:1-tile0 resnet50:int8_qdq:1 mobilenet_v3_large:int8_qdq:1 bert_base:int8_dyn:1 minilm_l6:int8_dyn:1 whisper_small:encoder_int8_dyn:1 whisper_small:decoder_int8_dyn:1"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out="$2"; shift 2;; --models) rows="$2"; shift 2;; --cases) cases="$2"; shift 2;; --every) every="$2"; shift 2;;
    --aot) aot=1; shift;; --merge) merge+=(--merge "$2"); shift 2;; --markdown) md="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
mkdir -p "$out"
dotnet publish -c Release "$here/GoldenModels/GoldenModels.csproj" -o "$out/jit" -v q -clp:NoSummary || exit 1
args=""
for spec in $rows; do
  read -r model_file ref_dir < <("$ONEMODEL_PY" "$here/model_path.py" "$spec" 2>/dev/null) || continue
  [ -f "$model_file" ] && [ -d "$ref_dir/inputs" ] || { echo "skip $spec (not fetched)" >&2; continue; }
  args="$args model $(echo "$spec" | tr ':/' '__')=$model_file=$ref_dir"
done
[ -d "$cases" ] && args="$args --cases $cases --every $every"
aotargs=()
if [ "$aot" = 1 ]; then
  dotnet publish -c Release "$here/GoldenModels/GoldenModels.csproj" -p:PublishAot=true -o "$out/aot" -v q -clp:NoSummary || exit 1
  aotargs=(--aot-cmd "$out/aot/GoldenModels $args")
fi
mdargs=(); [ -n "$md" ] && mdargs=(--markdown "$md")
tag="$(uname -s | tr A-Z a-z)-$(uname -m | tr A-Z a-z)"
"$ONEMODEL_PY" "$here/golden_hash.py" --cmd "dotnet $out/jit/GoldenModels.dll $args" ${aotargs[@]+"${aotargs[@]}"} --quiet --timeout 3000 \
  --save "$out/results-$tag.json" ${merge[@]+"${merge[@]}"} ${mdargs[@]+"${mdargs[@]}"}
