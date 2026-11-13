#!/usr/bin/env bash
# Installs the released ONNX Runtime packages the bar harness compares against, as black boxes (wheels installed and run, never opened),
# each in its own uv venv under $ONEMODEL_BAR_ORT (default $ONEMODEL_HOME/bar-ort) because the wheels share one module name.
#
#   setup_ort.sh webgpu|coreml|cuda|dml
#
# webgpu  onnxruntime + onnxruntime-ep-webgpu: the native WebGPU EP, Dawn over Vulkan (Linux), Metal (macOS) or D3D12 (Windows). It is shipped
#         as a plugin EP library registered at run time (https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html).
# coreml  onnxruntime: the macOS wheel includes the CoreML EP.
# cuda    onnxruntime-gpu with the CUDA and cuDNN libraries from PyPI: information only, never gated.
# dml     onnxruntime-directml: Windows.
# Versions are pinned below; ORT_VERSION and WEBGPU_EP_VERSION override them. The pins that ran are in every result file.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$here/env.sh"
what="${1:?usage: setup_ort.sh webgpu|coreml|cuda|dml}"
ORT_VERSION="${ORT_VERSION:-1.31.1}"
WEBGPU_EP_VERSION="${WEBGPU_EP_VERSION:-0.4.0}"
venv="${ONEMODEL_BAR_ORT:-$ONEMODEL_HOME/bar-ort}/$what"
if [ ! -x "$venv/bin/python" ]; then
  uv venv --python 3.13 "$venv"
  case "$what" in
    webgpu) uv pip install --python "$venv/bin/python" "onnxruntime==$ORT_VERSION" "onnxruntime-ep-webgpu==$WEBGPU_EP_VERSION" numpy ;;
    coreml) uv pip install --python "$venv/bin/python" "onnxruntime==$ORT_VERSION" numpy ;;
    cuda) uv pip install --python "$venv/bin/python" "onnxruntime-gpu[cuda,cudnn]==$ORT_VERSION" numpy ;;
    dml) uv pip install --python "$venv/bin/python" "onnxruntime-directml==$ORT_VERSION" numpy ;;
    *) echo "unknown provider $what" >&2; exit 2 ;;
  esac
fi
"$venv/bin/python" - <<'E'
import onnxruntime as ort
print("onnxruntime", ort.__version__, ort.get_available_providers())
E
echo "$venv/bin/python"
