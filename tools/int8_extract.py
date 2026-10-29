#!/usr/bin/env python3
"""Cuts the part of a model between its inputs and one tensor into its own model with a reference directory for `Bar check`
(ONNX Runtime ORT_ENABLE_ALL outputs), to localize a difference.

    int8_extract.py MODEL.onnx REFERENCE_DIR OUTDIR OUTPUT_TENSOR [--dq]

With --dq the output is the dequantized form of OUTPUT_TENSOR (a QuantizeLinear output), so it is a float tensor.
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
ap.add_argument("tensor")
ap.add_argument("--dq", action="store_true")
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "ref", "outputs"), exist_ok=True)
m0 = onnx.load(a.model)
inputs = [i.name for i in m0.graph.input if i.name not in {x.name for x in m0.graph.initializer}]
target = a.tensor
if target.startswith("@"):
    # @N: the quantized output of the Nth (from 0) Conv, Gemm or MatMul node in the graph's order
    heavy = [n for n in m0.graph.node if n.op_type in ("Conv", "Gemm", "MatMul")]
    q = next(n for n in m0.graph.node if n.op_type == "QuantizeLinear" and n.input[0] == heavy[int(target[1:])].output[0])
    a.tensor = target = q.output[0]
if a.dq:
    # the DequantizeLinear that reads this tensor
    dq = next(n for n in m0.graph.node if n.op_type == "DequantizeLinear" and n.input[0] == a.tensor)
    target = dq.output[0]
tmp = os.path.join(a.out, "cut.onnx")
onnx.utils.extract_model(a.model, tmp, inputs, [target])
shutil.copy(tmp, os.path.join(a.out, "model.onnx"))
shutil.copytree(os.path.join(a.ref, "inputs"), os.path.join(a.out, "ref", "inputs"), dirs_exist_ok=True)
feeds = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(os.path.join(a.ref, "inputs", "*.npy"))}
so = ort.SessionOptions()
so.intra_op_num_threads = 1
sess = ort.InferenceSession(tmp, so, providers=["CPUExecutionProvider"])
outs = sess.run(None, feeds)
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
outs_d = ort.InferenceSession(tmp, so, providers=["CPUExecutionProvider"]).run(None, feeds)
name = sess.get_outputs()[0].name
np.save(os.path.join(a.out, "ref", "outputs", name.replace("/", "_") + ".npy"), outs[0])
d = np.abs(outs[0] - outs_d[0])
print("ORT ALL vs DISABLE: differing elements", int((d > 0).sum()), "of", d.size, "max", float(d.max()))
