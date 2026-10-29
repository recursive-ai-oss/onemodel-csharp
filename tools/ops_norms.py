#!/usr/bin/env python3
"""Generates randomized one-node cases for the normalisation, softmax, reduction, arg-reduction and CumSum operators, runs ONNX
Runtime on them as a black box, and writes model, inputs and expected outputs for the C# test (OracleNormsTests).

Usage: ops_norms.py OUTDIR [--seed N]

Each case is a directory with model.onnx, input_<i>.bin, output_<i>.bin (little-endian raw tensors) and meta.json listing names,
element types and shapes. The generator is seeded, so the same arguments give the same cases. A case that ONNX Runtime refuses to
load or run is listed in skipped.json with the reason instead of being written.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper as h

NP2T = {np.dtype("float32"): T.FLOAT, np.dtype("int64"): T.INT64, np.dtype("int32"): T.INT32}


def make_model(op, opset, inputs, initializers, out_dtypes, attrs, out_names=("y",)):
    """inputs: list of (name, array) fed at run time; initializers: list of (name, array) baked into the model."""
    node = h.make_node(op, [n for n, _ in inputs] + [n for n, _ in initializers], list(out_names), **attrs)
    graph = h.make_graph(
        [node],
        "g",
        [h.make_tensor_value_info(n, NP2T[a.dtype], list(a.shape)) for n, a in inputs],
        [h.make_tensor_value_info(n, NP2T[np.dtype(d)], None) for n, d in zip(out_names, out_dtypes)],
        initializer=[onnx.numpy_helper.from_array(a, n) for n, a in initializers],
    )
    m = h.make_model(graph, opset_imports=[h.make_opsetid("", opset)])
    m.ir_version = 9
    return m


class Writer:
    def __init__(self, outdir):
        self.outdir = outdir
        self.count = 0
        self.skipped = {}
        os.makedirs(outdir, exist_ok=True)

    def add(self, name, op, opset, inputs, initializers, out_dtypes, attrs, out_names=("y",)):
        model = make_model(op, opset, inputs, initializers, out_dtypes, attrs, out_names)
        try:
            sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            outs = sess.run(None, {n: a for n, a in inputs})
        except Exception as e:  # noqa: BLE001 - the reason is the record
            self.skipped[name] = str(e).splitlines()[0][:200]
            return
        d = os.path.join(self.outdir, name)
        os.makedirs(d, exist_ok=True)
        onnx.save(model, os.path.join(d, "model.onnx"))
        meta = {"op": op, "opset": opset, "inputs": [], "outputs": []}
        for i, (n, a) in enumerate(inputs):
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

    # LayerNormalization (opset 17): the shapes of transformer exports, odd widths, other axes, with and without bias.
    for i, (shape, axis) in enumerate([((2, 5, 384), -1), ((3, 7, 768), -1), ((4, 17), -1), ((2, 3, 5, 16), 2), ((6, 1), -1), ((2, 33, 3), 1)]):
        normalized = shape[axis if axis >= 0 else len(shape) + axis:]
        x = f32(*shape)
        s = f32(*normalized, lo=0.5, hi=1.5)
        b = f32(*normalized)
        for withb in (False, True):
            ins = [("x", x)]
            inits = [("s", s)] + ([("b", b)] if withb else [])
            w.add(f"layernorm_{i}_{int(withb)}", "LayerNormalization", 17, ins, inits, ["float32"] * 3, {"axis": axis, "epsilon": 1e-5},
                  out_names=("y", "mean", "inv"))
    # Instance / group / RMS normalisation.
    for i, shape in enumerate([(2, 4, 9), (1, 3, 8, 8), (3, 5, 1)]):
        c = shape[1]
        w.add(f"instancenorm_{i}", "InstanceNormalization", 6, [("x", f32(*shape))], [("s", f32(c, lo=0.5, hi=1.5)), ("b", f32(c))], ["float32"], {"epsilon": 1e-5})
    for opset, per in ((18, 2), (21, 6)):
        for i, shape in enumerate([(2, 6, 5), (1, 4, 3, 3)]):
            c = shape[1]
            n = per if opset == 18 else c
            if opset == 18:
                n = 2
            w.add(f"groupnorm_{opset}_{i}", "GroupNormalization", opset, [("x", f32(*shape))], [("s", f32(n, lo=0.5, hi=1.5)), ("b", f32(n))], ["float32"],
                  {"epsilon": 1e-5, "num_groups": 2})
    for i, shape in enumerate([(2, 5, 16), (3, 7)]):
        w.add(f"rmsnorm_{i}", "RMSNormalization", 23, [("x", f32(*shape))], [("s", f32(shape[-1], lo=0.5, hi=1.5))], ["float32"], {"epsilon": 1e-5})

    # Softmax family: old coerce-to-2D (11, 12) and per-axis (13, 17) semantics, every axis; large magnitudes and -inf entries.
    for op in ("Softmax", "LogSoftmax", "Hardmax"):
        for opset in (11, 13, 17):
            for i, shape in enumerate([(3, 4, 5), (2, 129), (7,), (2, 1, 6, 3)]):
                for axis in range(-len(shape), len(shape)):
                    if opset == 11 and axis < 0 and False:
                        continue
                    x = f32(*shape, lo=-20, hi=20)
                    w.add(f"{op.lower()}_{opset}_{i}_{axis}", op, opset, [("x", x)], [], ["float32"], {"axis": axis})
        x = f32(4, 9, lo=-5, hi=5)
        x[0, 3] = -np.inf
        x[1, :] = -np.inf
        x[1, 2] = 1.0
        w.add(f"{op.lower()}_neginf", op, 13, [("x", x)], [], ["float32"], {"axis": -1})

    # Reductions: every non-empty axis subset of a rank-4 shape and a few others, both axes forms, keepdims on and off.
    ops = ["ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp"]
    for op in ops:
        for shape in [(2, 3, 4, 5), (5, 130), (1, 6, 1), (3, 257)]:
            x = rng.uniform(0.5, 1.5, size=shape).astype(np.float32)
            for mask in range(1, 1 << len(shape)):
                axes = [i if rng.integers(2) else i - len(shape) for i in range(len(shape)) if mask >> i & 1]
                for keep in (0, 1):
                    for opset in (11, 13, 18):
                        axes_input = opset >= 13 if op == "ReduceSum" else opset >= 18
                        attrs = {"keepdims": keep}
                        inits = []
                        if axes_input:
                            inits.append(("axes", np.array(axes, np.int64)))
                        else:
                            attrs["axes"] = axes
                        if rng.random() < 0.35 or opset == 18 and mask == 1:
                            w.add(f"{op.lower()}_{opset}_{'x'.join(map(str, shape))}_{mask}_{keep}", op, opset, [("x", x)], inits, ["float32"], attrs)
        # All axes when none are given, and the empty-axes forms.
        x = rng.uniform(0.5, 1.5, size=(3, 4, 5)).astype(np.float32)
        w.add(f"{op.lower()}_all_11", op, 11, [("x", x)], [], ["float32"], {"keepdims": 1})
        w.add(f"{op.lower()}_all_18", op, 18, [("x", x)], [], ["float32"], {"keepdims": 0})
        xn = rng.uniform(-1.5, 1.5, size=(3, 4, 5)).astype(np.float32)
        w.add(f"{op.lower()}_noop_18", op, 18, [("x", xn)], [("axes", np.array([], np.int64))], ["float32"], {"noop_with_empty_axes": 1})
    # NaN / infinity inputs. Max and min are left out for NaN: ONNX Runtime's result depends on its internal code path there.
    xs = np.array([[1, np.nan, 3], [np.nan] * 3, [2, 5, 4], [np.inf, -np.inf, 1]], np.float32)
    xinf = np.array([[1, np.inf, 3], [-np.inf] * 3, [2, 5, 4], [np.inf, -np.inf, 1]], np.float32)
    for op in ["ReduceMax", "ReduceMin"]:
        for axis in (0, 1):
            w.add(f"{op.lower()}_inf_{axis}", op, 11, [("x", xinf)], [], ["float32"], {"axes": [axis], "keepdims": 0})
    for op in ["ReduceSum", "ReduceMean", "ReduceProd", "ReduceL1", "ReduceL2", "ReduceSumSquare"]:
        for axis in (0, 1):
            w.add(f"{op.lower()}_special_{axis}", op, 11, [("x", xs)], [], ["float32"], {"axes": [axis], "keepdims": 0})
    # Integer reductions.
    xi = rng.integers(-5, 6, size=(3, 4, 5)).astype(np.int64)
    for op in ["ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd"]:
        w.add(f"{op.lower()}_int64", op, 13 if op != "ReduceSum" else 11, [("x", xi)], [], ["int64"], {"axes": [1], "keepdims": 0})

    # ArgMax / ArgMin with ties and NaN.
    for op in ("ArgMax", "ArgMin"):
        for opset in (11, 12, 13):
            for i, shape in enumerate([(3, 4, 5), (7,), (2, 33), (4, 1, 6)]):
                # Few distinct values, so ties are common. No NaN: ONNX Runtime's choice depends on its internal code path.
                x = rng.integers(0, 4, size=shape).astype(np.float32)
                for axis in range(-len(shape), len(shape)):
                    for keep in (0, 1):
                        for last in ((0, 1) if opset >= 12 else (0,)):
                            attrs = {"axis": axis, "keepdims": keep}
                            if opset >= 12:
                                attrs["select_last_index"] = last
                            if rng.random() < 0.4:
                                w.add(f"{op.lower()}_{opset}_{i}_{axis}_{keep}_{last}", op, opset, [("x", x)], [], ["int64"], attrs)

    # CumSum.
    for opset in (11, 14):
        for i, shape in enumerate([(3, 4, 5), (9,), (2, 70)]):
            x = f32(*shape)
            for axis in range(-len(shape), len(shape)):
                for ex, rev in itertools.product((0, 1), repeat=2):
                    if rng.random() < 0.5:
                        w.add(f"cumsum_{opset}_{i}_{axis}_{ex}{rev}", "CumSum", opset, [("x", x)], [("axis", np.array(axis, np.int64))], ["float32"],
                              {"exclusive": ex, "reverse": rev})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("outdir")
    p.add_argument("--seed", type=int, default=20261029)
    a = p.parse_args()
    w = Writer(a.outdir)
    gen(w, np.random.default_rng(a.seed))
    with open(os.path.join(a.outdir, "skipped.json"), "w") as f:
        json.dump(w.skipped, f, indent=1, sort_keys=True)
    print(f"wrote {w.count} cases, skipped {len(w.skipped)}; onnxruntime {ort.__version__}, onnx {onnx.__version__}")


if __name__ == "__main__":
    sys.exit(main())
