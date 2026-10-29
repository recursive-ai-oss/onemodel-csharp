#!/usr/bin/env bash
# Creates (or refreshes) the pinned oracle venv at $ONEMODEL_VENVS/oracles. Idempotent; works on
# Linux x64 and macOS arm64. Installs uv under $ONEMODEL_BIN when it is missing. No sudo.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv into $ONEMODEL_BIN"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$ONEMODEL_BIN" INSTALLER_NO_MODIFY_PATH=1 sh
fi

venv="$ONEMODEL_VENVS/oracles"
[ -x "$venv/bin/python" ] || uv venv --python 3.13 "$venv"

pip() { uv pip install --python "$venv/bin/python" "$@"; }

if [ "$(uname -s)" = Linux ]; then
  # The default Linux wheels bundle CUDA; the CPU index is smaller and is what we compare against.
  pip --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match "torch==2.15.0" "torchvision==0.29.2"
fi
pip -r "$here/requirements-oracles.txt"

"$venv/bin/python" - <<'EOF'
import importlib.metadata as m
for p in ("onnxruntime", "onnx", "onnxscript", "torch", "torchvision", "transformers", "numpy"):
    print(f"{p}=={m.version(p)}")
EOF
uv pip freeze --python "$venv/bin/python" > "$venv/freeze.txt"
echo "oracle venv ready: $venv (full freeze in $venv/freeze.txt)"
