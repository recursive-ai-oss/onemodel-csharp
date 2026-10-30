#!/usr/bin/env bash
# Converts reference models to the OneModel format and checks the result against the ONNX file:
#   * the outputs of Bar check (fast mode, same threads) are bit-identical for the ONNX file and for the converted file;
#   * with --py, tools/format_check.py (Python, numpy reader) finds every initializer byte-equal;
#   * load time (Bar load) and peak RSS (/usr/bin/time around Bar check) are printed for both.
#
#   format_compare.sh [--rows "minilm_l6:fp32:1 ..."] [--threads 1] [--out DIR] [--py]
#
# Row specs are model:variant:batch as in check_reference.sh. Not a publishable timing run: niced, stops when a benchmark lock
# appears (tools/light.sh).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
rows="resnet50:fp32:1 mobilenet_v3_large:fp32:1 bert_base:fp32:1 minilm_l6:fp32:1 whisper_small:encoder_fp32:1 whisper_small:decoder_fp32:1 crnn:fp32:1-val-00000539 dbnet:fp32:1-tile0 resnet50:int8_qdq:1 mobilenet_v3_large:int8_qdq:1 bert_base:int8_dyn:1 minilm_l6:int8_dyn:1"
threads=1; out="${ONEMODEL_HOME}/format"; py=0
while [ $# -gt 0 ]; do
  case "$1" in
    --rows) rows="$2"; shift 2;; --threads) threads="$2"; shift 2;; --out) out="$2"; shift 2;; --py) py=1; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
mkdir -p "$out"
if [ "$(uname -s)" = Darwin ]; then tm=(/usr/bin/time -l -o); else tm=(/usr/bin/time -v -o); fi
rss() { # file -> peak RSS in MB
  if [ "$(uname -s)" = Darwin ]; then awk '/maximum resident set size/{printf "%.0f", $1/1048576}' "$1"
  else awk '/Maximum resident set size/{printf "%.0f", $6/1024}' "$1"; fi
}

"$here/light.sh" dotnet build -c Release "$here/../bench/Bar/Bar.csproj" -o "$out/bin" -v q -clp:NoSummary || exit 1
bar="$out/bin/Bar.dll"
fail=0
printf '%-44s %-8s %9s %9s %9s %9s %8s %8s\n' row identical onnx_MB ours_MB onnx_load ours_load onnx_rss ours_rss
for spec in $rows; do
  read -r model_file ref_dir < <("$ONEMODEL_PY" "$here/model_path.py" "$spec")
  tag="$(echo "$spec" | tr ':/' '__')"
  d="$out/$tag"; rm -rf "$d"; mkdir -p "$d"
  conv="$d/model.onemodel"
  "$here/light.sh" dotnet "$bar" convert --model "$model_file" --out "$conv" >"$d/convert.log" || { echo "$spec: convert failed"; fail=1; continue; }
  "${tm[@]}" "$d/rss_onnx.txt" "$here/light.sh" dotnet "$bar" check --model "$model_file" --ref "$ref_dir" --threads "$threads" --tol 1 --dump "$d/onnx" >"$d/check_onnx.log"
  "${tm[@]}" "$d/rss_ours.txt" "$here/light.sh" dotnet "$bar" check --model "$conv" --ref "$ref_dir" --threads "$threads" --tol 1 --dump "$d/fmt" >"$d/check_fmt.log"
  same=yes
  for f in "$d"/onnx/*.f32; do cmp -s "$f" "$d/fmt/$(basename "$f")" || same=NO; done
  [ -n "$(ls "$d"/onnx 2>/dev/null)" ] || same=NO
  [ "$same" = yes ] || fail=1
  lo=$("$here/light.sh" dotnet "$bar" load --model "$model_file" --threads "$threads" | sed 's/.*median of [0-9]* later //')
  lf=$("$here/light.sh" dotnet "$bar" load --model "$conv" --threads "$threads" | sed 's/.*median of [0-9]* later //')
  printf '%-44s %-8s %9s %9s %9s %9s %8s %8s\n' "$spec" "$same" \
    "$(( $(stat -c %s "$model_file" 2>/dev/null || stat -f %z "$model_file") / 1048576 ))" \
    "$(( $(stat -c %s "$conv" 2>/dev/null || stat -f %z "$conv") / 1048576 ))" "$lo" "$lf" "$(rss "$d/rss_onnx.txt")" "$(rss "$d/rss_ours.txt")"
  if [ "$py" = 1 ]; then
    "$here/light.sh" "$ONEMODEL_PY" "$here/format_check.py" "$model_file" "$conv" || fail=1
  fi
  rm -rf "$d/onnx" "$d/fmt"
  [ "${KEEP_CONVERTED:-0}" = 1 ] || rm -f "$conv"
done
exit $fail
