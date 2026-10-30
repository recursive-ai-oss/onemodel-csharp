#!/usr/bin/env bash
# Runs the int8 metric (docs/oracles.md) on reference rows: Bar check with --dump, then check_int8.py.
#
#   check_int8.sh [--rows "resnet50:int8_qdq:1 bert_base:int8_dyn:32 ..."] [--threads N] [--out DIR]
#
# Row specs are model:variant:batch as in check_reference.sh. Not a timing run (niced, stops when a benchmark lock appears).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
rows="resnet50:int8_qdq:1 resnet50:int8_qdq:8 mobilenet_v3_large:int8_qdq:1 mobilenet_v3_large:int8_qdq:8 bert_base:int8_dyn:1 bert_base:int8_dyn:32 minilm_l6:int8_dyn:1 minilm_l6:int8_dyn:32 whisper_small:decoder_int8_dyn:1 whisper_small:encoder_int8_dyn:1"
threads=1; out="${ONEMODEL_HOME}/check_int8"
while [ $# -gt 0 ]; do
  case "$1" in --rows) rows="$2";; --threads) threads="$2";; --out) out="$2";; *) echo "unknown option $1" >&2; exit 2;; esac
  shift 2
done
mkdir -p "$out"
"$here/light.sh" dotnet build -c Release "$here/../bench/Bar/Bar.csproj" -o "$out/bin" -v q -clp:NoSummary || exit 1
fail=0
for spec in $rows; do
  read -r model_file ref_dir < <("$ONEMODEL_PY" "$here/model_path.py" "$spec")
  case "$spec" in resnet50:*|mobilenet*) task=class;; *decoder*|crnn:*) task=token;; *) task=none;; esac
  d="$out/$(echo "$spec" | tr ':/' '__')"; rm -rf "$d"; mkdir -p "$d"
  echo "== $spec (task $task, $threads threads)"
  "$here/light.sh" dotnet "$out/bin/Bar.dll" check --model "$model_file" --ref "$ref_dir" --threads "$threads" --tol 1 --dump "$d" | grep -E 'hash|check'
  "$here/light.sh" "$ONEMODEL_PY" "$here/check_int8.py" "$model_file" "$ref_dir" "$d" --task "$task" --threads "$threads" || fail=1
done
exit $fail
