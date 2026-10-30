#!/usr/bin/env python3
"""Taps for dynamically quantized models: adds the scale output of every DynamicQuantizeLinear and the (float-cast) int32 result
of every MatMulInteger as extra outputs, with ONNX Runtime's output for them, so the first product where we leave ONNX Runtime
can be found. Same arguments and use as int8_taps.py. The int32 sums must agree exactly when their inputs do.
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
ap.add_argument("--level", default="DISABLE")
a = ap.parse_args()

m = onnx.load(a.model)
k = 0
for n in list(m.graph.node):
    if n.op_type == "DynamicQuantizeLinear":
        name = f"tap_{k:03d}_scale"
        m.graph.node.append(h.make_node("Identity", [n.output[1]], [name]))
    elif n.op_type == "MatMulInteger":
        name = f"tap_{k:03d}_mm"
        c = f"{name}_c"
        m.graph.node.append(h.make_node("Cast", [n.output[0]], [name], to=onnx.TensorProto.FLOAT))
    else:
        continue
    m.graph.output.append(h.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    k += 1
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
print(k, "taps")
