#!/usr/bin/env python3
"""Applies the int8 metric of docs/oracles.md to one model row.

    check_int8.py MODEL.onnx REFERENCE_DIR OURS_DIR [--task none|class|token] [--threads 1]

REFERENCE_DIR holds inputs/*.npy and outputs/*.npy (ONNX Runtime ORT_ENABLE_ALL, from make_reference.py); OURS_DIR holds one
<output>.f32 per output, written by `Bar check --dump OURS_DIR`. The statistics of an output are

  * the 1st-percentile cosine similarity over the rows of the last axis (the minimum when there are fewer than 100 rows);
  * for rank >= 3 outputs, the minimum cosine of the mean-pooled vectors (mean over the second-to-last axis);
  * the decision agreement: rows whose argmax over the last axis is the same as the reference's.

The floor for each is what ONNX Runtime itself shows against its own ORT_ENABLE_ALL output: ORT_DISABLE_ALL, or, when those two
outputs are bit-identical (the float kernels do not depend on the level), the worst of four runs of ORT_ENABLE_ALL with relative
noise of 1e-7 on the float inputs. A row passes when, for every float output,

  (a) the 1st-percentile row cosine and (rank >= 3) the pooled cosine are each at least min(0.99, their floor); and
  (b) with --task class or token (class per image, token per position): decisions agree on at least 95% of the positions, or
      on at least as many as the floor run agrees; a single decision (one image) must agree.

Scaled error and the floor run's scaled error are printed, not gated. Argmax over a hidden dimension is not a decision, so
--task none (encoders) has no part (b).
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


def lowcos(x, y):
    x = x.reshape(-1, x.shape[-1]).astype(np.float64)
    y = y.reshape(-1, y.shape[-1]).astype(np.float64)
    c = (x * y).sum(1) / np.maximum(np.sqrt((x * x).sum(1) * (y * y).sum(1)), 1e-30)
    return float(c.min() if len(c) < 100 else np.percentile(c, 1))


def metrics(x, y):
    """(1st-percentile row cosine, min pooled cosine or None, decisions agreeing, decisions)."""
    row = lowcos(x, y)
    pooled = lowcos(x.mean(-2), y.mean(-2)) if x.ndim >= 3 else None
    ax, ay = x.reshape(-1, x.shape[-1]).argmax(1), y.reshape(-1, y.shape[-1]).argmax(1)
    return row, pooled, int((ax == ay).sum()), len(ax)


feeds = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(os.path.join(a.ref, "inputs", "*.npy"))}
def session(level):
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    so.intra_op_num_threads = a.threads
    return ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])


sess = session(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
names = [o.name for o in sess.get_outputs()]
floor_outs = [dict(zip(names, sess.run(None, feeds)))]
refs = {n: np.load(os.path.join(a.ref, "outputs", n + ".npy")) for n in names if os.path.exists(os.path.join(a.ref, "outputs", n + ".npy"))}
floor_kind = "ORT_DISABLE_ALL"
if all(np.array_equal(floor_outs[0][n], refs[n]) for n in refs):
    # The two levels are bit-identical: use ONNX Runtime's response to 1e-7 relative noise on the float inputs instead.
    floor_kind = "1e-7 input noise"
    rng = np.random.default_rng(1)
    full = session(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    floor_outs = []
    for _ in range(4):
        f2 = {k: (v * (1 + 1e-7 * rng.standard_normal(v.shape)).astype(np.float32) if v.dtype == np.float32 else v) for k, v in feeds.items()}
        floor_outs.append(dict(zip(names, full.run(None, f2))))
print(f"floor: {floor_kind}")

ok = True
for n in names:
    f = os.path.join(a.ref, "outputs", n + ".npy")
    if not os.path.exists(f) or not np.issubdtype(floor_outs[0][n].dtype, np.floating):
        continue
    ref = refs[n]
    ours = np.fromfile(os.path.join(a.ours, n + ".f32"), np.float32).reshape(ref.shape)
    o_row, o_pool, o_agree, count = metrics(ours, ref)
    fl = [metrics(o[n], ref) for o in floor_outs]
    s_row = min(m[0] for m in fl)
    s_pool = None if fl[0][1] is None else min(m[1] for m in fl)
    s_agree = min(m[2] for m in fl)
    scaled = float(np.abs(ours - ref).max() / max(np.abs(ref).max(), 1e-30))
    ort_scaled = float(max(np.abs(o[n] - ref).max() for o in floor_outs) / max(np.abs(ref).max(), 1e-30))
    good = o_row >= min(0.99, s_row)
    if o_pool is not None:
        good &= o_pool >= min(0.99, s_pool)
    if a.task != "none":
        good &= o_agree == count if count == 1 else (o_agree >= 0.95 * count or o_agree >= s_agree)
    ok &= good
    pool = "" if o_pool is None else f" pooled cos {o_pool:.4f} (floor {s_pool:.4f})"
    print(f"{n} {list(ref.shape)}: {'PASS' if good else 'FAIL'}  row cos p1 {o_row:.4f} (floor {s_row:.4f}){pool}  "
          f"decisions {o_agree}/{count} (floor {s_agree}/{count})  [scaled {scaled:.2e}, floor run {ort_scaled:.2e}]")
sys.exit(0 if ok else 1)
