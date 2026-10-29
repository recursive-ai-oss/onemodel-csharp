"""Fetch the ONNX backend-test data directories from the pinned release and count the cases.

    fetch_backend_test.py [--record]

Only members under onnx/backend/test/data/ are extracted from the release tag archive; nothing
else in the archive is opened. Output: <cache>/onnx-backend-test/{node,pytorch-converted,
pytorch-operator,simple}/<case>/ (model.onnx plus test_data_set_*/ with .pb inputs and outputs).
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

from common import BACKEND_TEST, CACHE, load_manifest, log, save_manifest, sha256_file

PREFIX = "onnx/backend/test/data/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    man = load_manifest()
    bt = man["backendTest"]
    arc = CACHE / "dl" / bt["archiveUrl"].rsplit("/", 1)[-1]
    if not arc.exists():
        arc.parent.mkdir(parents=True, exist_ok=True)
        log("downloading", bt["archiveUrl"])
        urllib.request.urlretrieve(bt["archiveUrl"], arc)
    h = sha256_file(arc)
    if not bt["archiveSha256"]:
        bt["archiveSha256"] = h  # first fetch; written back with --record
    elif h != bt["archiveSha256"]:
        raise SystemExit(f"archive hash mismatch: {h}")

    done = BACKEND_TEST / ".extracted"
    if not done.exists():
        with tarfile.open(arc) as t:
            for m in t:
                i = m.name.find(PREFIX)
                if i < 0 or not m.isfile():
                    continue
                rel = m.name[i + len(PREFIX):]
                parts = Path(rel).parts
                if not rel or ".." in parts or rel.startswith("/"):
                    raise SystemExit(f"unsafe archive member {m.name}")
                dst = BACKEND_TEST / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as src, open(dst, "wb") as out:
                    out.write(src.read())
        done.write_text(h + "\n")

    counts = {}
    for sub in sorted(p for p in BACKEND_TEST.iterdir() if p.is_dir()):
        cases = [c for c in sub.iterdir() if c.is_dir() or c.suffix == ".onnx"]
        with_model = [c for c in cases if c.suffix == ".onnx" or (c / "model.onnx").exists()]
        counts[sub.name] = {"cases": len(cases), "withModel": len(with_model)}
    bt["counts"] = counts
    for k, v in counts.items():
        log(f"{k}: {v['cases']} cases ({v['withModel']} with model.onnx)")
    if a.record:
        save_manifest(man)


if __name__ == "__main__":
    sys.exit(main())
