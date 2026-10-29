#!/usr/bin/env python3
"""Adds the dequantized activations of a QDQ model as extra outputs and writes a reference directory for `Bar check`, so the first
layer where our result leaves ONNX Runtime's can be found.

    int8_taps.py MODEL.onnx REFERENCE_DIR OUTDIR [--every N] [--level ALL|DISABLE]

Writes OUTDIR/model.onnx and OUTDIR/ref/{inputs,outputs}. Tap k is the output of the k*N-th DequantizeLinear whose input is not a
constant. Run: Bar check --model OUTDIR/model.onnx --ref OUTDIR/ref --threads 1
"""
import argparse
import glob
import os
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("ref")
ap.add_argument("out")
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--level", default="ALL")
a = ap.parse_args()

m = onnx.load(a.model)
init = {i.name for i in m.graph.initializer}
dqs = [n for n in m.graph.node if n.op_type == "DequantizeLinear" and n.input[0] not in init]
taps = dqs[:: a.every]
for k, n in enumerate(taps):
    m.graph.node.append(h.make_node("Identity", [n.output[0]], [f"tap_{k:03d}"]))
    m.graph.output.append(h.make_tensor_value_info(f"tap_{k:03d}", onnx.TensorProto.FLOAT, None))
os.makedirs(os.path.join(a.out, "ref", "outputs"), exist_ok=True)
onnx.save(m, os.path.join(a.out, "model.onnx"))
shutil.copytree(os.path.join(a.ref, "inputs"), os.path.join(a.out, "ref", "inputs"), dirs_exist_ok=True)
feeds = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(os.path.join(a.ref, "inputs", "*.npy"))}
so = ort.SessionOptions()
so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, f"ORT_{'ENABLE_ALL' if a.level == 'ALL' else 'DISABLE_ALL'}")
so.intra_op_num_threads = 1
sess = ort.InferenceSession(os.path.join(a.out, "model.onnx"), so, providers=["CPUExecutionProvider"])
for o, v in zip(sess.get_outputs(), sess.run(None, feeds)):
    np.save(os.path.join(a.out, "ref", "outputs", o.name + ".npy"), v)
print(len(taps), "taps")
