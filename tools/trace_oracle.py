#!/usr/bin/env python3
"""Runs the ONNX files written by the tracing tests (TraceDump, ONEMODEL_TRACE_DUMP=<dir>) in ONNX Runtime as a black box and compares
with the eager outputs the tests saved. Each model has two cases: the shape it was traced at ("a") and a different shape ("b"), which
checks that symbolic axes really are symbolic.

usage: trace_oracle.py <dir> [--tol 1e-4]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    root = Path(args.dir)
    failed = 0
    for model_path in sorted(root.glob("*.onnx")):
        name = model_path.stem
        ops = Counter(n.op_type for n in onnx.load(model_path).graph.node)
        print(f"{name}: {sum(ops.values())} nodes: " + ", ".join(f"{k}x{v}" for k, v in sorted(ops.items())))
        for level in (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, ort.GraphOptimizationLevel.ORT_DISABLE_ALL):
            opts = ort.SessionOptions()
            opts.graph_optimization_level = level
            sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
            for tag in ("a", "b"):
                meta = json.loads((root / f"{name}.{tag}.json").read_text())
                blob = (root / f"{name}.{tag}.in.bin").read_bytes()
                feeds = {}
                for i in meta["inputs"]:
                    raw = blob[i["offset"]:i["offset"] + i["bytes"]]
                    feeds[i["name"]] = np.frombuffer(raw, dtype=i["dtype"]).reshape(i["shape"])
                want = np.fromfile(root / f"{name}.{tag}.out.bin", dtype=np.float32).reshape(meta["outputShape"])
                got = sess.run([meta["output"]], feeds)[0]
                err = float(np.max(np.abs(got - want)) / max(float(np.max(np.abs(want))), 1e-30))
                ok = got.shape == want.shape and err <= args.tol
                failed += not ok
                print(f"  {name} {tag} ORT {level.name}: scaled max error {err:.3g} {'ok' if ok else 'FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
