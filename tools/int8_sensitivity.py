#!/usr/bin/env python3
"""Sensitivity of a quantized model to float noise: runs ONNX Runtime on the reference inputs and on inputs with relative noise
of a few ulp, and prints the scaled difference and the per-row decision agreement of the outputs.

    int8_sensitivity.py MODEL.onnx REFERENCE_DIR [REL_NOISE=1e-7] [TRIALS=5]
"""
import glob
import os
import sys

import numpy as np
import onnxruntime as ort

model, ref = sys.argv[1:3]
rel = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-7
trials = int(sys.argv[4]) if len(sys.argv) > 4 else 5
feeds = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(os.path.join(ref, "inputs", "*.npy"))}
so = ort.SessionOptions()
so.intra_op_num_threads = 1
sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
base = sess.run(None, feeds)
rng = np.random.default_rng(1)
for t in range(trials):
    f2 = {k: (v * (1 + rel * rng.standard_normal(v.shape)).astype(np.float32) if v.dtype == np.float32 else v) for k, v in feeds.items()}
    out = sess.run(None, f2)
    for b, o in zip(base, out):
        scaled = np.abs(o - b).max() / np.abs(b).max()
        agree = (o.argmax(-1) == b.argmax(-1)).sum() if o.ndim >= 2 else 0
        x, y = o.reshape(-1, o.shape[-1]).astype(np.float64), b.reshape(-1, b.shape[-1]).astype(np.float64)
        cos = ((x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))).min()
        print(f"trial {t}: scaled {scaled:.3e} min row cos {cos:.4f} top1 {agree}/{o.shape[-2] if o.ndim >= 2 else 1}")
