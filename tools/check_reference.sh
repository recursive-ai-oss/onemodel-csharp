#!/usr/bin/env bash
# Runs the library on the reference models and compares with the stored ONNX Runtime outputs (see docs/reference-models.md).
#
#   check_reference.sh [--rows "minilm_l6:fp32:1 resnet50:fp32:1 ..."] [--threads "1 N"] [--out DIR]
#
# For every row and thread count it runs `Bar check` (per-output scaled and element-wise error, output hashes of two runs),
# then reports whether the output hashes at the different thread counts are equal. "N" means every core. Not a timing run:
# no lock is taken, the work is niced, and it stops if a benchmark lock appears (tools/light.sh).
# Paths come from ONEMODEL_HOME (tools/env.sh).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"

rows="resnet50:fp32:1 mobilenet_v3_large:fp32:1 bert_base:fp32:1 minilm_l6:fp32:1 whisper_small:encoder_fp32:1 whisper_small:decoder_fp32:1 crnn:fp32:1-val-00000539 dbnet:fp32:1-tile0"
threads="1 N"; out="${ONEMODEL_HOME}/check"; mode=fast
while [ $# -gt 0 ]; do
  case "$1" in
    --rows) rows="$2";; --threads) threads="$2";; --out) out="$2";; --mode) mode="$2";;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
  shift 2
done
mkdir -p "$out"
if [ "$(uname -s)" = Darwin ]; then cores="$(sysctl -n hw.ncpu)"; else cores="$(nproc)"; fi

"$here/light.sh" dotnet build -c Release "$here/../bench/Bar/Bar.csproj" -o "$out/bin" -v q -clp:NoSummary || exit 1
bar="$out/bin/Bar.dll"

fail=0
for spec in $rows; do
  read -r model_file ref_dir < <("$ONEMODEL_PY" "$here/model_path.py" "$spec")
  echo "== $spec"
  prev=""
  for t in $threads; do
    n="$t"; [ "$t" = N ] && n="$cores"
    log="$out/$(echo "$spec" | tr ':/' '__').t$t.log"
    "$here/light.sh" dotnet "$bar" check --model "$model_file" --ref "$ref_dir" --threads "$n" --mode "$mode" | tee "$log"
    [ "${PIPESTATUS[0]}" = 0 ] || fail=1
    sig="$(grep -o 'hash [0-9A-F]*' "$log" | sort | tr '\n' ' ')"
    if [ -n "$prev" ] && [ "$sig" != "$prev" ]; then echo "  HASHES DIFFER between thread counts"; fail=1; fi
    prev="$sig"
  done
done
exit $fail
