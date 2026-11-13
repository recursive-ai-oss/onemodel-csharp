#!/usr/bin/env bash
# Classifies the rows of an earlier bar run again without timing anything: reads where each configuration's placement log put the nodes (a
# session is created per configuration, no model is run), applies the current gate-row rule, and writes DST as a copy of SRC with the new
# classes in DST/ref/*.json and a new DST/bar-PLATFORM.{jsonl,md}.
#
#   reclassify.sh SRC DST --platform linux|macos --ref webgpu|coreml,webgpu
#
# SRC/ref/ROW.json (first provider) and SRC/ref/NAME-ROW.json (the others) are read; the cuda- and ortcpu- files are copied as they are.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../env.sh"
src="$1"; dst="$2"; shift 2
platform=""; ref=""
while [ $# -gt 0 ]; do case "$1" in --platform) platform="$2"; shift 2;; --ref) ref="$2"; shift 2;; *) echo "unknown option $1" >&2; exit 2;; esac; done
[ -n "$platform" ] && [ -n "$ref" ] || { echo "usage: reclassify.sh SRC DST --platform P --ref R" >&2; exit 2; }
ortbase="${ONEMODEL_BAR_ORT:-$ONEMODEL_HOME/bar-ort}"
mkdir -p "$dst"
rsync -a --exclude cases --exclude gpureport --exclude ort-tmp "$src/" "$dst/"
for f in "$dst"/ref/*.json; do
  case "$(basename "$f")" in cuda-*|ortcpu-*) continue;; esac
  prov="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['provider'])" "$f")"
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/reclass.XXXXXX")"
  TMPDIR="$tmp" "$ortbase/$prov/bin/python" "$here/ort_run.py" --reclassify "$f" --out "$f.new" 2>&1 | sed "s/^/[$(basename "$f")] /"
  [ -f "$f.new" ] && mv "$f.new" "$f"
  rm -rf "$tmp"
done
python3 "$here/collect.py" "$dst" --platform "$platform" --ref "$ref"
