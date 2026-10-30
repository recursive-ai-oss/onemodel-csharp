#!/usr/bin/env bash
# Indicative cost of exact mode: best warm time per model, fast versus exact, at the given thread counts. Not a publishable benchmark
# (it takes no benchmark lock); use bench/Bar for those.   golden_cost.sh ["1 8"] [reps]
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
threads="${1:-1 8}"; reps="${2:-5}"
rows="resnet50:fp32:1 mobilenet_v3_large:fp32:1 bert_base:fp32:1 minilm_l6:fp32:1 whisper_small:encoder_fp32:1 whisper_small:decoder_fp32:1 crnn:fp32:1-val-00000539 dbnet:fp32:1-tile0 resnet50:int8_qdq:1 mobilenet_v3_large:int8_qdq:1 bert_base:int8_dyn:1 minilm_l6:int8_dyn:1 whisper_small:encoder_int8_dyn:1 whisper_small:decoder_int8_dyn:1"
out="${ONEMODEL_HOME}/golden"; mkdir -p "$out"
dotnet publish -c Release "$here/GoldenModels/GoldenModels.csproj" -o "$out/jit" -v q -clp:NoSummary || exit 1
args=()
for spec in $rows; do
  read -r f r < <("$ONEMODEL_PY" "$here/model_path.py" "$spec" 2>/dev/null) || continue
  [ -f "$f" ] && [ -d "$r/inputs" ] && args+=(model "$(echo "$spec" | tr ':/' '__')=$f=$r")
done
for t in $threads; do for mode in fast exact; do
  dotnet "$out/jit/GoldenModels.dll" --mode $mode --threads "$t" --time "$reps" "${args[@]}" 2>&1 >/dev/null | grep '^time '
done; done
