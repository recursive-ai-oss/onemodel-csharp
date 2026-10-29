"""Generates the small graphs under testdata/graphs and the ONNX Runtime reference outputs for them.

    make_graphs.py [--out DIR]

Everything is seeded; the same command writes the same bytes. The graphs are ours (nothing is taken from another project), and the
reference outputs are what ONNX Runtime computes for them, with its version recorded in the JSON. Needs the pinned oracle
environment (tools/setup-oracles.sh).

mlp.onnx    x[N,16] -> Gemm(transB) -> Relu -> MatMul -> Add -> Mul(Constant) -> Reshape(Constant) -> Identity, output [N,2,4].
            Uses every operator of the first operator set: Add, Mul, Relu, MatMul, Gemm, Reshape, Identity, Constant.
chain.onnx  x[N,64] through ten Add/Mul/Relu steps and a branch, to exercise buffer reuse.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

SEED = 20261029
OPSET = 17


def mlp(rng: np.random.Generator) -> onnx.ModelProto:
    w1 = rng.standard_normal((32, 16)).astype(np.float32) * 0.3
    b1 = rng.standard_normal(32).astype(np.float32) * 0.1
    w2 = rng.standard_normal((32, 8)).astype(np.float32) * 0.3
    b2 = rng.standard_normal(8).astype(np.float32) * 0.1
    nodes = [
        helper.make_node("Gemm", ["x", "w1", "b1"], ["h"], transB=1, alpha=1.0, beta=1.0),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node("MatMul", ["r", "w2"], ["m"]),
        helper.make_node("Add", ["m", "b2"], ["a"]),
        helper.make_node("Constant", [], ["scale"], value=numpy_helper.from_array(np.array([0.5], np.float32), "scale")),
        helper.make_node("Mul", ["a", "scale"], ["s"]),
        helper.make_node("Constant", [], ["target"], value=numpy_helper.from_array(np.array([-1, 2, 4], np.int64), "target")),
        helper.make_node("Reshape", ["s", "target"], ["y0"]),
        helper.make_node("Identity", ["y0"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 2, 4])],
        [numpy_helper.from_array(w1, "w1"), numpy_helper.from_array(b1, "b1"),
         numpy_helper.from_array(w2, "w2"), numpy_helper.from_array(b2, "b2")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=8)


def chain(rng: np.random.Generator) -> onnx.ModelProto:
    inits, nodes = [], []
    prev = "x"
    for i in range(10):
        k = f"k{i}"
        op = ["Add", "Mul", "Relu"][i % 3]
        if op != "Relu":
            inits.append(numpy_helper.from_array(rng.standard_normal(64).astype(np.float32) * 0.5 + 1.0, k))
        out = f"t{i}"
        nodes.append(helper.make_node(op, [prev] if op == "Relu" else [prev, k], [out]))
        prev = out
    nodes.append(helper.make_node("Add", [prev, "t2"], ["y"]))  # t2 stays alive across seven steps
    graph = helper.make_graph(
        nodes, "chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 64])],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "testdata" / "graphs"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(SEED))
    report = {"onnxruntime": ort.__version__, "onnx": onnx.__version__, "seed": SEED, "opset": OPSET, "cases": {}}
    for name, build, width in (("mlp", mlp, 16), ("chain", chain, 64)):
        model = build(rng)
        onnx.checker.check_model(model)
        path = out / f"{name}.onnx"
        path.write_bytes(model.SerializeToString())
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for batch in (1, 5):
            x = rng.standard_normal((batch, width)).astype(np.float32)
            (y,) = sess.run(None, {"x": x})
            report["cases"][f"{name}-{batch}"] = {
                "model": f"{name}.onnx", "shape": list(x.shape), "input": x.ravel().tolist(),
                "outShape": list(y.shape), "output": y.ravel().tolist(),
            }
    (out / "reference.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
