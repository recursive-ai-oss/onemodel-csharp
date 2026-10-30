#!/usr/bin/env bash
# Runs the golden-hash app on the browser-wasm runtime under node (headless) and saves its results for tools/golden_hash.py --merge.
#
#   golden_wasm.sh [--aot] [--out DIR] [--models "crnn:fp32:1-val-00000539 ..."] [--cases DIR] [--every N]
#
# Without --aot the app runs on the interpreter; with --aot it is compiled ahead of time (needs the wasm-tools workload).
# NODE names the node binary (default: node on PATH). The interpreter is slow, so the default model set is the small models and
# the default case sample is every 8th node case; the other rows run a superset and the table compares the names both have.
# The wasm runtime has no host file access: the model files, inputs and cases are copied into its in-memory file system.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
out="$ONEMODEL_HOME/golden"; cases="$ONEMODEL_CACHE/onnx-backend-test/node"; every=8; aot=0; node="${NODE:-node}"
rows="crnn:fp32:1-val-00000539 dbnet:fp32:1-tile0 minilm_l6:fp32:1 minilm_l6:int8_dyn:1"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out="$2"; shift 2;; --models) rows="$2"; shift 2;; --cases) cases="$2"; shift 2;; --every) every="$2"; shift 2;;
    --aot) aot=1; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
kind=interp; aotprop=(); [ "$aot" = 1 ] && { kind=aot; aotprop=(-p:RunAOTCompilation=true); }
mkdir -p "$out"
dotnet publish -c Release "$here/../tests/WasmSmoke/WasmSmoke.csproj" ${aotprop[@]+"${aotprop[@]}"} -o "$out/wasm-$kind" -v q -clp:NoSummary || exit 1
www="$out/wasm-$kind/wwwroot"
echo '{"type":"module"}' > "$www/package.json"   # lets node load dotnet.js as an ES module
copy=(); args=""
for spec in $rows; do
  read -r model_file ref_dir < <("$ONEMODEL_PY" "$here/model_path.py" "$spec" 2>/dev/null) || continue
  [ -f "$model_file" ] && [ -d "$ref_dir/inputs" ] || { echo "skip $spec (not fetched)" >&2; continue; }
  copy+=(--copy "$model_file" --copy "$ref_dir/inputs")
  # An external-data model keeps its weights next to the file.
  [ -d "$model_file.data" ] && copy+=(--copy "$model_file.data")
  args="$args model $(echo "$spec" | tr ':/' '__')=$model_file=$ref_dir"
done
[ -d "$cases" ] && { copy+=(--copy "$cases"); args="$args --cases $cases --every $every"; }
tag="$(uname -s | tr A-Z a-z)-wasm"
# shellcheck disable=SC2086
"$ONEMODEL_PY" "$here/golden_hash.py" --no-builtin --quiet --timeout 7200 --tag "$tag" --save "$out/results-$tag-$kind.json" \
  --row "wasm-$kind::$node $here/../tests/WasmSmoke/main.mjs $www ${copy[*]} -- $args"
