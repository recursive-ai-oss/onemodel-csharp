#!/usr/bin/env bash
# Runs cases through onnxruntime-web's WebGPU execution provider in headless Chrome, as a black box (the released npm package, loaded and
# run, never opened), so the browser rows of tools/gpu_browser.sh have a number to be compared with.
#
#   ort_web.sh --out DIR --adapter nvidia|swiftshader [--runs N] ROW...
#
# Reads DIR/cases/ROW/case.bin (written by gpu_browser.sh) and writes DIR/ort/ADAPTER-ROW.json. The package is installed once into
# $ONEMODEL_HOME/ortweb (ORT_WEB_VERSION picks a version; default: the latest release); the version that ran is in every result.
# Needs the same environment as gpu_browser.sh: NODE, CHROME_GPU_FLAGS.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
out=""; adapter=nvidia; runs=10
while [ $# -gt 0 ]; do
  case "$1" in --out) out="$2"; shift 2;; --adapter) adapter="$2"; shift 2;; --runs) runs="$2"; shift 2;; *) break;; esac
done
node="${NODE:-node}"; npm="$(dirname "$(command -v "$node")")/npm"
dir="$ONEMODEL_HOME/ortweb"
if [ ! -d "$dir/node_modules/onnxruntime-web" ]; then
  mkdir -p "$dir" && (cd "$dir" && "$npm" install --no-audit --no-fund "onnxruntime-web@${ORT_WEB_VERSION:-latest}" > "$dir/install.log" 2>&1) || { echo "npm install failed, see $dir/install.log" >&2; exit 1; }
fi
declare -A MODEL
MODEL[minilm_fp32_b1]=minilm_l6/minilm_l6.onnx; MODEL[minilm_fp32_b8]=minilm_l6/minilm_l6.onnx; MODEL[minilm_fp16_b1]=minilm_l6/minilm_l6_fp16.onnx
MODEL[minilm_int8_b1]=minilm_l6/minilm_l6_int8_dyn.onnx; MODEL[bert_fp32_b1]=bert_base/bert_base.onnx; MODEL[bert_fp32_b8]=bert_base/bert_base.onnx
MODEL[bert_int8_b1]=bert_base/bert_base_int8_dyn.onnx; MODEL[resnet50_fp32_b1]=resnet50/resnet50.onnx; MODEL[resnet50_fp32_b8]=resnet50/resnet50.onnx
MODEL[resnet50_fp16_b1]=resnet50/resnet50_fp16.onnx; MODEL[resnet50_int8_b1]=resnet50/resnet50_int8_qdq.onnx
MODEL[mobilenet_fp32_b1]=mobilenet_v3_large/mobilenet_v3_large.onnx; MODEL[mobilenet_fp32_b8]=mobilenet_v3_large/mobilenet_v3_large.onnx
MODEL[mobilenet_int8_b1]=mobilenet_v3_large/mobilenet_v3_large_int8_qdq.onnx; MODEL[dbnet_fp32_b1]=dbnet/dbnet_lite_v1.onnx; MODEL[crnn_fp32_b1]=crnn/crnn_v0.onnx
MODEL[whisper_enc_b1]=whisper_small/whisper_small_encoder.onnx; MODEL[whisper_dec_b1]=whisper_small/whisper_small_decoder.onnx
MODEL[whisper_enc_int8_b1]=whisper_small/whisper_small_encoder_int8_dyn.onnx; MODEL[whisper_dec_int8_b1]=whisper_small/whisper_small_decoder_int8_dyn.onnx
mkdir -p "$out/ort"
for r in "$@"; do
  [ -n "${MODEL[$r]:-}" ] || { echo "ort: no model for row $r" >&2; continue; }
  # A fresh browser per row; the page serves the harness page, the package, the models and the cases.
  "$node" "$here/web/run-page.mjs" --root "$here/web/ort" --mounts "/ort=$dir/node_modules/onnxruntime-web/dist,/models=$ONEMODEL_CACHE/models,/cases=$out/cases" \
    --path "/index.html?model=/models/${MODEL[$r]}&case=/cases/$r/case.bin&runs=$runs&warmups=3&label=$r" \
    --timeout 1500 --out "$out/ort/$adapter-$r.json" --log 1 2> "$out/ort/$adapter-$r.log" > /dev/null || echo "ort $adapter $r: no result" >&2
done
