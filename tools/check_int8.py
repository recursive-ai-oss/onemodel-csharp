#!/usr/bin/env python3
"""Applies the int8 metric of docs/oracles.md to one model row.

    check_int8.py MODEL.onnx REFERENCE_DIR OURS_DIR [--task none|class|token] [--threads 1]

REFERENCE_DIR holds inputs/*.npy and outputs/*.npy (ONNX Runtime ORT_ENABLE_ALL, from make_reference.py); OURS_DIR holds one
<output>.f32 per output, written by `Bar check --dump OURS_DIR`. The script runs the model once more at ORT_DISABLE_ALL to find
ONNX Runtime's own spread between its two levels. A row passes when, for every float output,

  (a) the minimum cosine similarity of ours against ORT_ENABLE_ALL, per row of the last axis and (for rank >= 3 outputs) per
      mean-pooled vector over the second-to-last axis, is at least min(0.99, the same minimum for ORT_DISABLE_ALL against
      ORT_ENABLE_ALL); and
  (b) with --task class or token (class per image, token per position): the decisions (argmax over the last axis) agree on at
      least 95% of the positions, or at least as often as ORT's two levels agree with each other; a single decision (one image)
      must agree.

Scaled error and ORT's own difference are printed, not gated. Argmax over a hidden dimension is not a decision, so --task none
(encoders) has no part (b).
"""
import argparse
import glob
import os
import sys

import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("ref")
ap.add_argument("ours")
ap.add_argument("--task", default="none", choices=["none", "class", "token"])
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()


def mincos(x, y):
    x = x.reshape(-1, x.shape[-1]).astype(np.float64)
    y = y.reshape(-1, y.shape[-1]).astype(np.float64)
    c = (x * y).sum(1) / np.maximum(np.sqrt((x * x).sum(1) * (y * y).sum(1)), 1e-30)
    return float(c.min())


def metrics(x, y):
    """(min row cosine, min pooled cosine or None, decisions agreeing, decisions)."""
    row = mincos(x, y)
    pooled = mincos(x.mean(-2), y.mean(-2)) if x.ndim >= 3 else None
    ax, ay = x.reshape(-1, x.shape[-1]).argmax(1), y.reshape(-1, y.shape[-1]).argmax(1)
    return row, pooled, int((ax == ay).sum()), len(ax)


feeds = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(os.path.join(a.ref, "inputs", "*.npy"))}
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
so.intra_op_num_threads = a.threads
sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
names = [o.name for o in sess.get_outputs()]
disabled = dict(zip(names, sess.run(None, feeds)))

ok = True
for n in names:
    f = os.path.join(a.ref, "outputs", n + ".npy")
    if not os.path.exists(f) or not np.issubdtype(disabled[n].dtype, np.floating):
        continue
    ref = np.load(f)
    ours = np.fromfile(os.path.join(a.ours, n + ".f32"), np.float32).reshape(ref.shape)
    o_row, o_pool, o_agree, count = metrics(ours, ref)
    s_row, s_pool, s_agree, _ = metrics(disabled[n], ref)
    scaled = float(np.abs(ours - ref).max() / max(np.abs(ref).max(), 1e-30))
    ort_scaled = float(np.abs(disabled[n] - ref).max() / max(np.abs(ref).max(), 1e-30))
    good = o_row >= min(0.99, s_row)
    if o_pool is not None:
        good &= o_pool >= min(0.99, s_pool)
    if a.task != "none":
        good &= o_agree == count if count == 1 else (o_agree >= 0.95 * count or o_agree >= s_agree)
    ok &= good
    pool = "" if o_pool is None else f" pooled cos {o_pool:.4f} (ORT {s_pool:.4f})"
    print(f"{n} {list(ref.shape)}: {'PASS' if good else 'FAIL'}  min row cos {o_row:.4f} (ORT {s_row:.4f}){pool}  "
          f"decisions {o_agree}/{count} (ORT {s_agree}/{count})  [scaled {scaled:.2e}, ORT self {ort_scaled:.2e}]")
sys.exit(0 if ok else 1)
