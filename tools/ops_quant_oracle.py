#!/usr/bin/env python3
"""Generates randomized one-node cases for the 8-bit quantization operators, runs ONNX Runtime on them as a black box, and writes
model, inputs and expected outputs for the C# test (OracleQuantTests).

Usage: ops_quant_oracle.py OUTDIR [--seed N]

Operators: QuantizeLinear, DequantizeLinear, DynamicQuantizeLinear, MatMulInteger, QLinearMatMul, ConvInteger, QLinearConv.
Each case is a directory with model.onnx, input_<i>.bin, output_<i>.bin (little-endian raw tensors) and meta.json. The generator is
seeded, so the same arguments give the same cases. A case ONNX Runtime refuses is listed in skipped.json with the reason.
"""
import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper as h

NP2T = {np.dtype("float32"): T.FLOAT, np.dtype("int32"): T.INT32, np.dtype("uint8"): T.UINT8, np.dtype("int8"): T.INT8,
        np.dtype("uint16"): T.UINT16, np.dtype("int16"): T.INT16}


class Writer:
    def __init__(self, outdir):
        self.outdir = outdir
        self.count = 0
        self.skipped = {}
        os.makedirs(outdir, exist_ok=True)

    def add(self, name, op, opset, inputs, initializers, out_dtypes, attrs, out_names=("y",), tolerance=0):
        """inputs: fed at run time; initializers: baked into the model. Both are (name, array); an array of None is an omitted input.
        tolerance: allowed difference in the last place for 8-bit outputs (0 = exact)."""
        order = inputs + initializers
        node = h.make_node(op, [n if a is not None else "" for n, a in order], list(out_names), **attrs)
        real = [(n, a) for n, a in order if a is not None]
        graph = h.make_graph(
            [node], "g",
            [h.make_tensor_value_info(n, NP2T[a.dtype], list(a.shape)) for n, a in inputs if a is not None],
            [h.make_tensor_value_info(n, NP2T[np.dtype(d)], None) for n, d in zip(out_names, out_dtypes)],
            initializer=[onnx.numpy_helper.from_array(a, n) for n, a in initializers if a is not None])
        m = h.make_model(graph, opset_imports=[h.make_opsetid("", opset)])
        m.ir_version = 9
        feeds = {n: a for n, a in inputs if a is not None}
        try:
            sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
            outs = sess.run(None, feeds)
        except Exception as e:  # noqa: BLE001 - the reason is the record
            self.skipped[name] = str(e).splitlines()[0][:200]
            return
        d = os.path.join(self.outdir, name)
        os.makedirs(d, exist_ok=True)
        onnx.save(m, os.path.join(d, "model.onnx"))
        meta = {"op": op, "opset": opset, "tolerance": tolerance, "inputs": [], "outputs": []}
        for i, (n, a) in enumerate([(n, a) for n, a in inputs if a is not None]):
            a.astype(a.dtype.newbyteorder("<")).tofile(os.path.join(d, f"input_{i}.bin"))
            meta["inputs"].append({"name": n, "dtype": str(a.dtype), "shape": list(a.shape)})
        for i, o in enumerate(outs):
            np.ascontiguousarray(o).tofile(os.path.join(d, f"output_{i}.bin"))
            meta["outputs"].append({"dtype": str(o.dtype), "shape": list(o.shape)})
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f)
        self.count += 1


def gen(w, rng):
    f32 = lambda *s, lo=-3.0, hi=3.0: rng.uniform(lo, hi, size=s).astype(np.float32)  # noqa: E731

    def q(dt, *s):
        info = np.iinfo(dt)
        return rng.integers(info.min, info.max + 1, size=s).astype(dt)

    def zp(dt, *s):
        info = np.iinfo(dt)
        return rng.integers(info.min, info.max + 1, size=s).astype(dt)

    # --- QuantizeLinear ---
    for opset in (10, 13, 19):
        for dt in (np.uint8, np.int8):
            for withzp in (False, True):
                if not withzp and dt == np.int8:
                    continue
                s = np.float32(0.0173)
                x = f32(4, 7, 9, lo=-5, hi=5)
                # exact ties after division: (k + 0.5) * scale is not exact in float, so also use a power-of-two scale
                w.add(f"quantize_{opset}_{np.dtype(dt).name}_{int(withzp)}", "QuantizeLinear", opset, [("x", x)],
                      [("s", np.array(s)), ("z", zp(dt))] if withzp else [("s", np.array(s))], [dt], {})
        # power-of-two scale: x/scale is exact, so half-way values test round-half-even
        x = (rng.integers(-300, 300, size=(3, 5, 8)) * 0.5).astype(np.float32) * np.float32(0.125)
        w.add(f"quantize_{opset}_ties", "QuantizeLinear", opset, [("x", x)], [("s", np.array(0.125, np.float32)), ("z", zp(np.uint8))], [np.uint8], {})
        xs = np.array([np.nan, np.inf, -np.inf, 1e30, -1e30, 0.0, -0.0, 1.0], np.float32)
        w.add(f"quantize_{opset}_special", "QuantizeLinear", opset, [("x", xs)], [("s", np.array(0.5, np.float32)), ("z", np.array(3, np.uint8))], [np.uint8], {})
    for opset in (13, 19):
        for axis in (0, 1, -1):
            x = f32(3, 5, 6, lo=-4, hi=4)
            n = x.shape[axis]
            sc = rng.uniform(0.01, 0.1, size=n).astype(np.float32)
            for dt in (np.uint8, np.int8):
                w.add(f"quantize_axis_{opset}_{axis}_{np.dtype(dt).name}", "QuantizeLinear", opset, [("x", x)],
                      [("s", sc), ("z", zp(dt, n))], [dt], {"axis": axis})
    x = f32(4, 4, lo=-2, hi=2)
    w.add("quantize_21_u16", "QuantizeLinear", 21, [("x", x)], [("s", np.array(0.001, np.float32)), ("z", zp(np.uint16))], [np.uint16], {})
    w.add("quantize_21_i16", "QuantizeLinear", 21, [("x", x)], [("s", np.array(0.001, np.float32)), ("z", zp(np.int16))], [np.int16], {})

    # Many elements, so a rare rounding disagreement shows up.
    for i, (dt, sc) in enumerate([(np.int8, 0.0186239), (np.uint8, 0.0113900788), (np.int8, 0.0437)]):
        x = f32(1, 64, 56, 56, lo=-3, hi=3) * np.float32(1.0)
        w.add(f"quantize_big_{i}", "QuantizeLinear", 13, [("x", x)], [("s", np.array(sc, np.float32)), ("z", zp(dt))], [dt], {})
    # DequantizeLinear and Add chains are not operators, but QuantizeLinear of sums is what the rewrite relies on.

    # A quantized residual Add, with the parameters of ResNet-50's first block (zero point -128 on the output clips like a Relu).
    for i, (sa, za, sb, zb, sy, zy) in enumerate([(0.011185410432517529, -14, 0.04366272687911987, 20, 0.01862390525639057, -128),
                                                  (0.0113900788128376, 14, 0.015234442427754402, -128, 0.012361573986709118, -128)]):
        shp = (1, 64, 56, 56)
        x = [q(np.int8, *shp) for _ in range(2)]
        w.add_model(f"qdq_add_{i}", qdq_unit("Add", [shp, shp], [sa, sb], [za, zb], sy, zy), {"q0": x[0], "q1": x[1]})

    # A dynamically quantized linear layer as quantize_dynamic writes it: DynamicQuantizeLinear, MatMulInteger, Cast, Mul by
    # input scale * weight scale, Add of the bias.
    for i, (shp, n, bias) in enumerate([((1, 384), 1536, True), ((128, 768), 768, True), ((2, 33, 70), 41, False), ((5, 3, 16), 8, True)]):
        k = shp[-1]
        wq = q(np.int8, k, n)
        wscale = rng.uniform(0.001, 0.01, size=n).astype(np.float32)
        nodes = [
            h.make_node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"]),
            h.make_node("MatMulInteger", ["xq", "wq", "xz", "wz"], ["acc"]),
            h.make_node("Cast", ["acc"], ["accf"], to=T.FLOAT),
            h.make_node("Mul", ["xs", "wscale"], ["scales"]),
            h.make_node("Mul", ["accf", "scales"], ["y0"]),
        ]
        inits = [onnx.numpy_helper.from_array(wq, "wq"), onnx.numpy_helper.from_array(np.zeros((), np.int8), "wz"),
                 onnx.numpy_helper.from_array(wscale, "wscale")]
        out = "y0"
        if bias:
            nodes.append(h.make_node("Add", ["y0", "b"], ["y"]))
            inits.append(onnx.numpy_helper.from_array(f32(n), "b"))
            out = "y"
        g = h.make_graph(nodes, "g", [h.make_tensor_value_info("x", T.FLOAT, list(shp))], [h.make_tensor_value_info(out, T.FLOAT, None)], initializer=inits)
        m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
        m.ir_version = 9
        w.add_model(f"dynlinear_{i}", m, {"x": f32(*shp, lo=-2, hi=3)})

    # --- DequantizeLinear ---
    for opset in (10, 13, 19):
        for dt in (np.uint8, np.int8):
            w.add(f"dequantize_{opset}_{np.dtype(dt).name}", "DequantizeLinear", opset, [("x", q(dt, 5, 33))],
                  [("s", np.array(0.0371, np.float32)), ("z", zp(dt))], [np.float32], {})
            w.add(f"dequantize_{opset}_{np.dtype(dt).name}_nozp", "DequantizeLinear", opset, [("x", q(dt, 3, 17))],
                  [("s", np.array(0.5, np.float32))], [np.float32], {})
    w.add("dequantize_i32", "DequantizeLinear", 13, [("x", rng.integers(-100000, 100000, size=(4, 9)).astype(np.int32))],
          [("s", np.array(0.01, np.float32))], [np.float32], {})
    for axis in (0, 1, -1):
        for dt in (np.uint8, np.int8):
            x = q(dt, 4, 6, 5)
            n = x.shape[axis]
            w.add(f"dequantize_axis_{axis}_{np.dtype(dt).name}", "DequantizeLinear", 13, [("x", x)],
                  [("s", rng.uniform(0.01, 0.1, size=n).astype(np.float32)), ("z", zp(dt, n))], [np.float32], {"axis": axis})
    w.add("dequantize_21_u16", "DequantizeLinear", 21, [("x", q(np.uint16, 3, 5))], [("s", np.array(0.01, np.float32)), ("z", zp(np.uint16))], [np.float32], {})

    # --- DynamicQuantizeLinear ---
    for i, x in enumerate([f32(3, 5, 17, lo=-4, hi=9), f32(4, 33, lo=0.5, hi=7), f32(2, 40, lo=-8, hi=-0.25), np.zeros((2, 8), np.float32),
                           f32(1, 5, lo=-1e-6, hi=1e-6), f32(8, 128, lo=-100, hi=100), f32(1, 1, lo=3, hi=3)]):
        w.add(f"dynquant_{i}", "DynamicQuantizeLinear", 11, [("x", x)], [], [np.uint8, np.float32, np.uint8], {},
              out_names=("y", "scale", "zp"))

    # --- MatMulInteger ---
    shapes = [((1, 1), (1, 1)), ((3, 5), (5, 7)), ((17, 64), (64, 33)), ((64, 513), (513, 40)), ((130, 300), (300, 70)), ((7, 1030), (1030, 5)),
              ((2, 5, 9), (9, 4)), ((2, 3, 4), (2, 4, 5)), ((3, 4), (2, 4, 5)), ((100, 768), (768, 96))]
    for si, (sa, sb) in enumerate(shapes):
        for adt, bdt in ((np.uint8, np.int8), (np.int8, np.int8), (np.uint8, np.uint8), (np.int8, np.uint8)):
            for variant in ("nozp", "scalar", "perchan"):
                if variant != "scalar" and (adt, bdt) not in ((np.uint8, np.int8), (np.int8, np.uint8)):
                    continue
                if si not in (0, 1, 2, 3, 4, 6, 7, 9) and variant != "nozp":
                    continue
                a, b = q(adt, *sa), q(bdt, *sb)
                inits = []
                n = sb[-1]
                if variant == "scalar":
                    inits = [("az", zp(adt)), ("bz", zp(bdt))]
                elif variant == "perchan":
                    inits = [("az", zp(adt)), ("bz", zp(bdt, n))]
                for bconst in (True, False):
                    if not bconst and si > 3:
                        continue
                    w.add_ordered(f"matmulinteger_{si}_{np.dtype(adt).name}_{np.dtype(bdt).name}_{variant}_{'c' if bconst else 'v'}",
                                  "MatMulInteger", 10, [("a", a), ("b", b)] + inits, {"a"} if bconst else {"a", "b"}, [np.int32], {})

    # --- QLinearMatMul ---
    for si, (sa, sb) in enumerate(shapes[:5] + shapes[6:8]):
        for adt, bdt, ydt in ((np.uint8, np.int8, np.uint8), (np.int8, np.int8, np.int8), (np.uint8, np.uint8, np.uint8)):
            for perchan in (False, True):
                a, b = q(adt, *sa), q(bdt, *sb)
                n = sb[-1]
                a_s = np.array(rng.uniform(0.005, 0.05), np.float32)
                b_s = rng.uniform(0.005, 0.05, size=n).astype(np.float32) if perchan else np.array(rng.uniform(0.005, 0.05), np.float32)
                y_s = np.array(rng.uniform(0.5, 4.0), np.float32)
                order_in = [("a", a), ("a_s", a_s), ("a_z", zp(adt)), ("b", b), ("b_s", b_s), ("b_z", zp(bdt, n) if perchan else zp(bdt)), ("y_s", y_s), ("y_z", zp(ydt))]
                w.add_ordered(f"qlinearmatmul_{si}_{np.dtype(adt).name}_{np.dtype(bdt).name}_{np.dtype(ydt).name}_{int(perchan)}", "QLinearMatMul", 10,
                              order_in, {"a"}, [ydt], {}, tolerance=1)

    # --- ConvInteger / QLinearConv ---
    convs = [
        # (x shape, w shape, attrs)
        ((1, 3, 9, 9), (4, 3, 3, 3), {"pads": [1, 1, 1, 1]}),
        ((2, 4, 8, 7), (6, 4, 3, 3), {"strides": [2, 2], "pads": [1, 1, 1, 1]}),
        ((1, 6, 8, 8), (6, 3, 3, 3), {"group": 2, "pads": [1, 1, 1, 1]}),
        ((2, 5, 10, 10), (5, 1, 3, 3), {"group": 5, "pads": [1, 1, 1, 1]}),
        ((1, 5, 11, 9), (5, 1, 5, 5), {"group": 5, "pads": [2, 2, 2, 2], "strides": [2, 2]}),
        ((2, 8, 6, 6), (16, 8, 1, 1), {}),
        ((1, 8, 7, 7), (4, 8, 1, 1), {"strides": [2, 2]}),
        ((1, 3, 16), (4, 3, 5), {"pads": [2, 2]}),
        ((1, 2, 9, 9), (3, 2, 3, 3), {"dilations": [2, 2], "pads": [2, 2, 2, 2]}),
        ((1, 16, 14, 14), (32, 16, 3, 3), {"pads": [1, 1, 1, 1]}),
        ((1, 3, 7, 7), (4, 3, 3, 3), {"auto_pad": "SAME_UPPER"}),
    ]
    for ci, (xs, ws, attrs) in enumerate(convs):
        for xdt, wdt in ((np.uint8, np.int8), (np.uint8, np.uint8), (np.int8, np.int8), (np.int8, np.uint8)):
            x, wt = q(xdt, *xs), q(wdt, *ws)
            m = ws[0]
            for variant in ("nozp", "zp"):
                if variant != "zp" and (xdt, wdt) not in ((np.uint8, np.int8), (np.uint8, np.uint8)):
                    continue
                order_in = [("x", x), ("w", wt)]
                if variant == "zp":
                    order_in += [("xz", zp(xdt)), ("wz", zp(wdt))]
                elif variant == "perchan":
                    order_in += [("xz", zp(xdt)), ("wz", zp(wdt, m))]
                w.add_ordered(f"convinteger_{ci}_{np.dtype(xdt).name}_{np.dtype(wdt).name}_{variant}", "ConvInteger", 10, order_in, {"x"}, [np.int32], attrs)
        # QLinearConv
        for xdt, wdt, ydt in ((np.uint8, np.int8, np.uint8), (np.int8, np.int8, np.int8), (np.uint8, np.uint8, np.uint8)):
            x, wt = q(xdt, *xs), q(wdt, *ws)
            m = ws[0]
            for perchan in (False, True):
                for withbias in (False, True):
                    x_s = np.array(rng.uniform(0.005, 0.05), np.float32)
                    w_s = rng.uniform(0.005, 0.05, size=m).astype(np.float32) if perchan else np.array(rng.uniform(0.005, 0.05), np.float32)
                    y_s = np.array(rng.uniform(0.5, 4.0), np.float32)
                    order_in = [("x", x), ("x_s", x_s), ("x_z", zp(xdt)), ("w", wt), ("w_s", w_s), ("w_z", zp(wdt, m) if perchan else zp(wdt)),
                                ("y_s", y_s), ("y_z", zp(ydt))]
                    if withbias:
                        order_in.append(("bias", rng.integers(-5000, 5000, size=m).astype(np.int32)))
                    w.add_ordered(f"qlinearconv_{ci}_{np.dtype(xdt).name}_{np.dtype(wdt).name}_{int(perchan)}_{int(withbias)}", "QLinearConv", 10,
                                  order_in, {"x"}, [ydt], attrs, tolerance=1)


def add_model(self, name, m, feeds, tolerance=0):
    """Adds a whole model: feeds is a dict of graph inputs; ONNX Runtime's reference is the default optimization level (ORT_ENABLE_ALL)."""
    try:
        sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        outs = sess.run(None, feeds)
    except Exception as e:  # noqa: BLE001
        self.skipped[name] = str(e).splitlines()[0][:200]
        return
    d = os.path.join(self.outdir, name)
    os.makedirs(d, exist_ok=True)
    onnx.save(m, os.path.join(d, "model.onnx"))
    meta = {"op": "model", "opset": 17, "tolerance": tolerance, "inputs": [], "outputs": []}
    for i, gi in enumerate(m.graph.input):
        a = feeds[gi.name]
        a.astype(a.dtype.newbyteorder("<")).tofile(os.path.join(d, f"input_{i}.bin"))
        meta["inputs"].append({"name": gi.name, "dtype": str(a.dtype), "shape": list(a.shape)})
    for i, o in enumerate(outs):
        np.ascontiguousarray(o).tofile(os.path.join(d, f"output_{i}.bin"))
        meta["outputs"].append({"dtype": str(o.dtype), "shape": list(o.shape)})
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    self.count += 1


Writer.add_model = add_model


def qdq_unit(op, shapes, sa, za, sy, zy, attrs=None, dt=np.int8, opset=17):
    """DequantizeLinear for each input, one float op, QuantizeLinear: the shape of one quantized layer in a QDQ model."""
    tp = T.INT8 if dt == np.int8 else T.UINT8
    nodes, inits, ins = [], [], []
    for i, shp in enumerate(shapes):
        nodes.append(h.make_node("DequantizeLinear", [f"q{i}", f"s{i}", f"z{i}"], [f"d{i}"]))
        inits += [onnx.numpy_helper.from_array(np.array(sa[i], np.float32), f"s{i}"), onnx.numpy_helper.from_array(np.array(za[i], dt), f"z{i}")]
        ins.append(h.make_tensor_value_info(f"q{i}", tp, list(shp)))
    nodes.append(h.make_node(op, [f"d{i}" for i in range(len(shapes))], ["o"], **(attrs or {})))
    nodes.append(h.make_node("QuantizeLinear", ["o", "sy", "zy"], ["y"]))
    inits += [onnx.numpy_helper.from_array(np.array(sy, np.float32), "sy"), onnx.numpy_helper.from_array(np.array(zy, dt), "zy")]
    g = h.make_graph(nodes, "g", ins, [h.make_tensor_value_info("y", tp, None)], initializer=inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", opset)])
    m.ir_version = 9
    return m


def add_ordered(self, name, op, opset, order_in, runtime_names, out_dtypes, attrs, out_names=("y",), tolerance=0):
    """Like add, but the node's inputs are given in spec order; names in runtime_names are graph inputs, the rest initializers."""
    node = h.make_node(op, [n for n, _ in order_in], list(out_names), **attrs)
    graph = h.make_graph(
        [node], "g",
        [h.make_tensor_value_info(n, NP2T[a.dtype], list(a.shape)) for n, a in order_in if n in runtime_names],
        [h.make_tensor_value_info(n, NP2T[np.dtype(d)], None) for n, d in zip(out_names, out_dtypes)],
        initializer=[onnx.numpy_helper.from_array(np.asarray(a), n) for n, a in order_in if n not in runtime_names])
    m = h.make_model(graph, opset_imports=[h.make_opsetid("", opset)])
    m.ir_version = 9
    runtime = [(n, a) for n, a in order_in if n in runtime_names]
    try:
        sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        outs = sess.run(None, {n: a for n, a in runtime})
    except Exception as e:  # noqa: BLE001
        self.skipped[name] = str(e).splitlines()[0][:200]
        return
    d = os.path.join(self.outdir, name)
    os.makedirs(d, exist_ok=True)
    onnx.save(m, os.path.join(d, "model.onnx"))
    meta = {"op": op, "opset": opset, "tolerance": tolerance, "inputs": [], "outputs": []}
    for i, (n, a) in enumerate(runtime):
        a.astype(a.dtype.newbyteorder("<")).tofile(os.path.join(d, f"input_{i}.bin"))
        meta["inputs"].append({"name": n, "dtype": str(a.dtype), "shape": list(a.shape)})
    for i, o in enumerate(outs):
        np.ascontiguousarray(o).tofile(os.path.join(d, f"output_{i}.bin"))
        meta["outputs"].append({"dtype": str(o.dtype), "shape": list(o.shape)})
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    self.count += 1


Writer.add_ordered = add_ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=20261029)
    a = ap.parse_args()
    w = Writer(a.outdir)
    gen(w, np.random.default_rng(a.seed))
    with open(os.path.join(a.outdir, "skipped.json"), "w") as f:
        json.dump(w.skipped, f, indent=1)
    print(f"{w.count} cases written, {len(w.skipped)} skipped (ONNX Runtime {ort.__version__})")


if __name__ == "__main__":
    main()
