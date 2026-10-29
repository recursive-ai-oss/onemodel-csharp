#!/usr/bin/env python3
"""How far apart are ONNX Runtime's optimization levels on an int8 model, and where does our output sit among them?

    int8_levels.py MODEL.onnx REFERENCE_DIR OURS_DIR

REFERENCE_DIR holds inputs/*.npy (make_reference.py); OURS_DIR holds <output>.f32 files written by `Bar check --dump`. Prints the
scaled error max|a-b|/max|b| between every pair of {DISABLE_ALL, ENABLE_BASIC, ENABLE_EXTENDED, ENABLE_ALL, ours}.
"""
import glob
import itertools
import os
import sys

import numpy as np
import onnxruntime as ort

model, ref, ours = sys.argv[1:4]
feeds = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(os.path.join(ref, "inputs", "*.npy"))}
levels = {"DISABLE": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
          "EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}
outs = {}
for name, lv in levels.items():
    so = ort.SessionOptions()
    so.graph_optimization_level = lv
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    outs[name] = {n: v for n, v in zip(names, sess.run(None, feeds))}
outs["ours"] = {}
for n in outs["ALL"]:
    f = os.path.join(ours, n + ".f32")
    if os.path.exists(f):
        outs["ours"][n] = np.fromfile(f, np.float32).reshape(outs["ALL"][n].shape)
for out in outs["ALL"]:
    print("output", out)
    for a, b in itertools.combinations(outs, 2):
        if out in outs[a] and out in outs[b]:
            x, y = outs[a][out], outs[b][out]
            print(f"  {a:8s} vs {b:8s}: {np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-30):.3e}")
