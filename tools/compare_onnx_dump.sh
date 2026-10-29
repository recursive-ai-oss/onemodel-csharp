#!/usr/bin/env bash
# Compares the importer's structural dump with the onnx package's, for the backend-test data and some model files.
#
#   compare_onnx_dump.sh <OnnxSweep.dll> <python> <backend-test-root> <outdir> [model.onnx ...]
#
# Needs the .NET runtime on PATH and a Python with the onnx package. Prints the number of differing files per set and the first
# differences; exits 1 if any differ.
set -u
dll=$1; py=$2; root=$3; out=$4; shift 4
here=$(cd "$(dirname "$0")" && pwd)
bad=0
mkdir -p "$out"
for set in "$root"/*/; do
  name=$(basename "$set")
  n=$(find "$set" -name model.onnx | wc -l)
  [ "$n" -gt 0 ] || continue
  dotnet "$dll" sweep "$set" "$out/cs-$name" > "$out/sweep-$name.txt" 2>&1
  "$py" "$here/onnx_dump.py" sweep "$set" "$out/py-$name" > "$out/py-$name.log" 2>&1
  d=$(diff -rq "$out/py-$name" "$out/cs-$name" | wc -l)
  echo "$name: $n models, $d differing dumps"
  [ "$d" -eq 0 ] || { bad=1; diff -r "$out/py-$name" "$out/cs-$name" | head -10; }
done
if [ $# -gt 0 ]; then
  dotnet "$dll" dump "$out/cs-files" "$@" > "$out/files-cs.log" 2>&1
  "$py" "$here/onnx_dump.py" dump "$out/py-files" "$@" > "$out/files-py.log" 2>&1
  d=$(diff -rq "$out/py-files" "$out/cs-files" | wc -l)
  echo "files: $# models, $d differing dumps"
  [ "$d" -eq 0 ] || { bad=1; diff -r "$out/py-files" "$out/cs-files" | head -10; }
fi
exit $bad
