"""Writes a wide MLP and the reference directory the bar runner expects, for timing the runtime on a model it fully supports.

    make_bar_mlp.py OUTDIR [--batch 32] [--width 1024] [--depth 4]

OUTDIR/mlp.onnx, OUTDIR/ref/inputs/x.npy, OUTDIR/ref/outputs/y.npy (computed by ONNX Runtime, for the harness's sanity check).
Uses Gemm, Relu, Add and MatMul only. Seeded; nothing is committed from this, the files are large.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "ref" / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "ref" / "outputs").mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(20261029))
    nodes, inits, prev = [], [], "x"
    for i in range(a.depth):
        w = (rng.standard_normal((a.width, a.width)) / np.sqrt(a.width)).astype(np.float32)
        b = (rng.standard_normal(a.width) * 0.1).astype(np.float32)
        inits += [numpy_helper.from_array(w, f"w{i}"), numpy_helper.from_array(b, f"b{i}")]
        if i % 2 == 0:
            nodes.append(helper.make_node("Gemm", [prev, f"w{i}", f"b{i}"], [f"g{i}"], transB=1))
        else:
            nodes.append(helper.make_node("MatMul", [prev, f"w{i}"], [f"m{i}"]))
            nodes.append(helper.make_node("Add", [f"m{i}", f"b{i}"], [f"g{i}"]))
        nodes.append(helper.make_node("Relu", [f"g{i}"], [f"r{i}"]))
        prev = f"r{i}"
    nodes.append(helper.make_node("Identity", [prev], ["y"]))
    g = helper.make_graph(nodes, "mlp", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [a.batch, a.width])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [a.batch, a.width])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    (out / "mlp.onnx").write_bytes(m.SerializeToString())
    x = rng.standard_normal((a.batch, a.width)).astype(np.float32)
    (y,) = ort.InferenceSession(str(out / "mlp.onnx"), providers=["CPUExecutionProvider"]).run(None, {"x": x})
    np.save(out / "ref" / "inputs" / "x.npy", x)
    np.save(out / "ref" / "outputs" / "y.npy", y)


if __name__ == "__main__":
    main()
