#!/usr/bin/env bash
# Regenerates docs/conformance.md: runs every onnx/backend/test case in fast and exact mode, then writes the table.
#
#   tools/conformance.sh [--out DIR] [--table docs/conformance.md]
#
# Needs the frozen data (tools/fetch_backend_test.py) under $ONEMODEL_CACHE/onnx-backend-test. The result files are
# DIR/conformance-fast.tsv and DIR/conformance-exact.tsv (columns: group, case, opset, ops, mode, status, reason).
# Heavy-ish (a few minutes at most): on a shared host run it through the job queue.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
out="$ONEMODEL_HOME/conformance"; table="$here/../docs/conformance.md"
while [ $# -gt 0 ]; do
  case "$1" in --out) out="$2";; --table) table="$2";; *) echo "unknown option $1" >&2; exit 2;; esac
  shift 2
done
mkdir -p "$out"
dotnet build -c Release "$here/../tests/OnnxSweep/OnnxSweep.csproj" -o "$out/bin" -v q -clp:NoSummary
dotnet "$out/bin/RecursiveAI.OneModel.OnnxSweep.dll" conformance "$ONEMODEL_CACHE/onnx-backend-test" "$out" both
python3 "$here/conformance_table.py" "$out" --out "$table"
