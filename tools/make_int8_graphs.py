#!/usr/bin/env python3
"""Writes the small quantized graphs the NativeAOT smoke app embeds: testdata/graphs/int8_dyn.onnx (dynamic quantization of a
two-layer MLP) and int8_qdq.onnx (static QDQ of a small convolution net), each with the input and the ONNX Runtime output in a
.json file. Seeded; ONNX Runtime is used as a black box (quantization tool and reference).

    make_int8_graphs.py [OUTDIR]        (default testdata/graphs)
"""
import json
import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "testdata", "graphs")
rng = np.random.default_rng(20261029)


def w(name, *shape, scale=0.3):
    return nh.from_array((rng.standard_normal(shape) * scale).astype(np.float32), name)


def save(model, path, x):
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    y = sess.run(None, {"x": x})[0]
    meta = {"onnxruntime": ort.__version__, "shape": list(x.shape), "input": x.ravel().tolist(), "output": y.ravel().tolist()}
    with open(path[:-5] + ".json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f)
        f.write("\n")


def dyn(tmp):
    g = h.make_graph(
        [h.make_node("MatMul", ["x", "w1"], ["a"]), h.make_node("Add", ["a", "b1"], ["b"]), h.make_node("Relu", ["b"], ["c"]),
         h.make_node("MatMul", ["c", "w2"], ["y"])],
        "mlp", [h.make_tensor_value_info("x", T.FLOAT, [4, 32])], [h.make_tensor_value_info("y", T.FLOAT, [4, 8])],
        [w("w1", 32, 48), w("b1", 48, scale=0.1), w("w2", 48, 8)])
    src = os.path.join(tmp, "dyn_src.onnx")
    onnx.save(h.make_model(g, opset_imports=[h.make_opsetid("", 17)], ir_version=10), src)
    dst = os.path.join(out, "int8_dyn.onnx")
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=True)
    save(onnx.load(dst), dst, rng.standard_normal((4, 32)).astype(np.float32))


def qdq(tmp):
    g = h.make_graph(
        [h.make_node("Conv", ["x", "w1", "b1"], ["a"], pads=[1, 1, 1, 1]), h.make_node("Relu", ["a"], ["b"]),
         h.make_node("Conv", ["b", "w2"], ["c"], strides=[2, 2], pads=[1, 1, 1, 1]), h.make_node("Relu", ["c"], ["d"]),
         h.make_node("GlobalAveragePool", ["d"], ["e"]), h.make_node("Flatten", ["e"], ["f"]),
         h.make_node("Gemm", ["f", "w3", "b3"], ["y"], transB=1)],
        "net", [h.make_tensor_value_info("x", T.FLOAT, [1, 3, 12, 12])], [h.make_tensor_value_info("y", T.FLOAT, [1, 10])],
        [w("w1", 8, 3, 3, 3), w("b1", 8, scale=0.1), w("w2", 16, 8, 3, 3), w("w3", 10, 16), w("b3", 10, scale=0.1)])
    src = os.path.join(tmp, "qdq_src.onnx")
    onnx.save(h.make_model(g, opset_imports=[h.make_opsetid("", 17)], ir_version=10), src)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"x": rng.standard_normal((1, 3, 12, 12)).astype(np.float32)} for _ in range(16)])

        def get_next(self):
            return next(self.it, None)

    dst = os.path.join(out, "int8_qdq.onnx")
    quantize_static(src, dst, Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True)
    save(onnx.load(dst), dst, rng.standard_normal((1, 3, 12, 12)).astype(np.float32))


with tempfile.TemporaryDirectory() as tmp:
    dyn(tmp)
    qdq(tmp)
