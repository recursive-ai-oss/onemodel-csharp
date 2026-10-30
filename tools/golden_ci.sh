#!/usr/bin/env bash
# Exact-mode check for CI: runs the committed test graphs in exact mode and compares every output hash with
# testdata/graphs/golden-exact.txt. The expected hashes are the same on every OS, CPU and runtime (that is the point of exact
# mode), so a mismatch on any runner fails the job.
#
#   golden_ci.sh jit|aot|wasm [WASM_WWWROOT]   run the already-built/published app of that kind (see ci.yml)
#   golden_ci.sh update                        rewrite the expected file from a JIT run on this machine
#
# jit: dotnet <out>/GoldenModels.dll    aot: <out>/GoldenModels    (APP names the program; GRAPH_ARGS may be overridden)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
g="$(cd "$here/../testdata/graphs" && pwd)"
expected="$g/golden-exact.txt"
args=(graph "mlp5=$g/mlp.onnx=$g/reference.json#mlp-5" graph "chain5=$g/chain.onnx=$g/reference.json#chain-5"
      graph "shape_ops=$g/shape_ops.onnx=$g/shape_ops.json" graph "int8_dyn=$g/int8_dyn.onnx=$g/int8_dyn.json"
      graph "int8_qdq=$g/int8_qdq.onnx=$g/int8_qdq.json")
kind="${1:?jit, aot, wasm or update}"
case "$kind" in
  jit|update) run=(dotnet "${APP:?APP is the path of GoldenModels.dll}") ;;
  aot) run=("${APP:?APP is the path of the GoldenModels executable}") ;;
  wasm) run=(node "$here/../tests/WasmSmoke/main.mjs" "${2:?wwwroot}" --copy "$g" --) ;;
  *) echo "unknown kind $kind" >&2; exit 2 ;;
esac
got="$("${run[@]}" "${args[@]}" | tr -d '\r' | grep -E '^[^ ]+ [0-9a-f]{64}$' | LC_ALL=C sort)"
if [ "$kind" = update ]; then printf '%s\n' "$got" > "$expected"; echo "wrote $expected"; exit 0; fi
if diff <(printf '%s\n' "$got") <(tr -d '\r' < "$expected"); then echo "golden hashes equal ($kind)"; else echo "golden hash mismatch ($kind)" >&2; exit 1; fi
