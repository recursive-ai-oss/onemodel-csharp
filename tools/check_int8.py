#!/usr/bin/env python3
"""Applies the int8 metric of docs/oracles.md to one model row.

    check_int8.py MODEL.onnx REFERENCE_DIR OURS_DIR [--task none|class|token] [--threads 1]

REFERENCE_DIR holds inputs/*.npy and outputs/*.npy (ONNX Runtime ORT_ENABLE_ALL, from make_reference.py); OURS_DIR holds one
<output>.f32 per output, written by `Bar check --dump OURS_DIR`. The statistics of an output are

  * the 1st-percentile cosine similarity over the rows of the last axis (the minimum when there are fewer than 100 rows);
  * for rank >= 3 outputs, the minimum cosine of the mean-pooled vectors (mean over the second-to-last axis);
  * the decision agreement: rows whose argmax over the last axis is the same as the reference's.

The floor of each statistic is the smallest of three values, all measured against ORT_ENABLE_ALL: ORT_DISABLE_ALL; the worst of
16 runs of ORT_ENABLE_ALL with relative noise of 1e-7 on the float inputs; and, for dynamically quantized models, the worst of
16 runs with the same noise on the float tensor that feeds the first DynamicQuantizeLinear (the embedding output of an encoder
whose inputs are token ids, which cannot be perturbed themselves). All are ONNX Runtime's own sensitivity; the worst is the
honest floor. The line also prints the earlier floor (without the third value) so the effect of the added runs is visible. A row passes when, for every float output,

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
import onnx
import onnxruntime as ort
from onnx import helper as h

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("ref")
ap.add_argument("ours")
ap.add_argument("--task", default="none", choices=["none", "class", "token"])
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--trials", type=int, default=16, help="noise runs per noise site (default 16, the gated value)")
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
# Also the worst of four ORT_ENABLE_ALL runs with 1e-7 relative noise on the float inputs; the floor is the worse of the two spreads.
rng = np.random.default_rng(1)
full = session(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
full_opts = ort.SessionOptions()
full_opts.intra_op_num_threads = a.threads
for _ in range(a.trials):
    f2 = {k: (v * (1 + 1e-7 * rng.standard_normal(v.shape)).astype(np.float32) if v.dtype == np.float32 else v) for k, v in feeds.items()}
    floor_outs.append(dict(zip(names, full.run(None, f2))))

# Noise on the float tensor feeding the first DynamicQuantizeLinear. A float input "noise_tap" is added to that tensor (adding
# exact zeros leaves the unperturbed result unchanged); the tensor's shape is learned from a run that exposes it.
tap_outs = []
m0 = onnx.load(a.model)
graph_inputs = {i.name for i in m0.graph.input}
tap = next((n.input[0] for n in m0.graph.node if n.op_type == "DynamicQuantizeLinear"), None)
if tap is not None and tap not in graph_inputs:
    probe = onnx.load(a.model)
    probe.graph.output.append(h.make_tensor_value_info(tap, onnx.TensorProto.FLOAT, None))
    base = ort.InferenceSession(probe.SerializeToString(), full_opts, providers=["CPUExecutionProvider"]).run([tap], feeds)[0]
    for node in m0.graph.node:
        for k, x in enumerate(node.input):
            if x == tap:
                node.input[k] = tap + "_noisy"
    m0.graph.input.append(h.make_tensor_value_info("noise_tap", onnx.TensorProto.FLOAT, None))
    # the Add goes right after the node that produces the tap, which keeps the node list topologically sorted
    at = max(i for i, node in enumerate(m0.graph.node) if tap in node.output) + 1
    m0.graph.node.insert(at, h.make_node("Add", [tap, "noise_tap"], [tap + "_noisy"]))
    noisy = ort.InferenceSession(m0.SerializeToString(), full_opts, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(2)
    for _ in range(a.trials):
        d = (base * 1e-7 * rng.standard_normal(base.shape)).astype(np.float32)
        tap_outs.append(dict(zip(names, noisy.run(names, dict(feeds, noise_tap=d)))))
    print(f"embedding-noise floor: noise on {tap} {list(base.shape)}")

ok = True
for n in names:
    f = os.path.join(a.ref, "outputs", n + ".npy")
    if not os.path.exists(f) or not np.issubdtype(floor_outs[0][n].dtype, np.floating):
        continue
    ref = refs[n]
    ours = np.fromfile(os.path.join(a.ours, n + ".f32"), np.float32).reshape(ref.shape)
    o_row, o_pool, o_agree, count = metrics(ours, ref)
    fl0 = [metrics(o[n], ref) for o in floor_outs]
    fl = fl0 + [metrics(o[n], ref) for o in tap_outs]
    s_row = min(m[0] for m in fl)
    s_pool = None if fl[0][1] is None else min(m[1] for m in fl)
    s_agree = min(m[2] for m in fl)
    o_row0 = min(m[0] for m in fl0)
    o_pool0 = None if fl0[0][1] is None else min(m[1] for m in fl0)
    o_agree0 = min(m[2] for m in fl0)
    scaled = float(np.abs(ours - ref).max() / max(np.abs(ref).max(), 1e-30))
    ort_scaled = float(max(np.abs(o[n] - ref).max() for o in floor_outs + tap_outs) / max(np.abs(ref).max(), 1e-30))
    good = o_row >= min(0.99, s_row)
    if o_pool is not None:
        good &= o_pool >= min(0.99, s_pool)
    if a.task != "none":
        good &= o_agree == count if count == 1 else (o_agree >= 0.95 * count or o_agree >= s_agree)
    ok &= good
    pool = "" if o_pool is None else f" pooled cos {o_pool:.4f} (floor {s_pool:.4f}, old {o_pool0:.4f})"
    print(f"{n} {list(ref.shape)}: {'PASS' if good else 'FAIL'}  row cos p1 {o_row:.4f} (floor {s_row:.4f}, old {o_row0:.4f}){pool}  "
          f"decisions {o_agree}/{count} (floor {s_agree}/{count}, old {o_agree0})  [scaled {scaled:.2e}, floor run {ort_scaled:.2e}]")
sys.exit(0 if ok else 1)
