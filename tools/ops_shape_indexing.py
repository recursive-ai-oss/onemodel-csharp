"""Reference cases for the shape and indexing operators, with ONNX Runtime's outputs as the expected values.

    ops_shape_indexing.py --out DIR [--seed N]

Writes DIR/<case>/model.onnx and DIR/<case>/test_data_set_0/{input_i,output_j}.pb in the same layout as onnx/backend/test, so the
C# runner that reads backend-test cases reads these too. A case ONNX Runtime refuses to run gets an `expect_error` file naming its
message instead of outputs: our runner then expects a SessionException (or, where we are more lenient, reports it as a difference).

The graphs are ours: one node per model, shapes and attributes as the reference models use them plus edge cases (empty and size-1
extents, negative axes and indices, huge slice bounds, NaN, infinity, ties). Seeded: the same command writes the same bytes.
ONNX Runtime is used as a black box through its public Python API (CPU, one thread, all optimizations on).
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

NP2ONNX = {
    np.dtype("float32"): TensorProto.FLOAT, np.dtype("float64"): TensorProto.DOUBLE, np.dtype("float16"): TensorProto.FLOAT16,
    np.dtype("int8"): TensorProto.INT8, np.dtype("int16"): TensorProto.INT16, np.dtype("int32"): TensorProto.INT32,
    np.dtype("int64"): TensorProto.INT64, np.dtype("uint8"): TensorProto.UINT8, np.dtype("bool"): TensorProto.BOOL,
    np.dtype("uint64"): TensorProto.UINT64, np.dtype("uint16"): TensorProto.UINT16, np.dtype("uint32"): TensorProto.UINT32,
}

I64 = np.int64
EDGE_INTS = [-(2**63), -100, -7, -3, -2, -1, 0, 1, 2, 3, 5, 9, 100, 2**31 - 1, 2**63 - 1]


class Cases:
    def __init__(self, out: Path, seed: int):
        self.out = out
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.errors = 0
        self.manifest = []

    def add(self, op, opset, inputs, attrs=None, out_types=None, consts=(), n_out=1, tag=""):
        """inputs: list of (array|None). consts: indices baked in as initializers. out_types: numpy dtypes of the outputs."""
        attrs = attrs or {}
        names, graph_in, inits, feeds = [], [], [], []
        for i, a in enumerate(inputs):
            if a is None:
                names.append("")
                continue
            n = f"i{i}"
            names.append(n)
            t = numpy_helper.from_array(a, n)
            if i in consts:
                inits.append(t)
            else:
                graph_in.append(helper.make_tensor_value_info(n, NP2ONNX[a.dtype], list(a.shape)))
                feeds.append((n, a))
        if out_types is None:
            out_types = [inputs[0].dtype]
        out_names = [f"o{j}" for j in range(n_out)]
        outs = [helper.make_tensor_value_info(n, NP2ONNX[np.dtype(t)], None) for n, t in zip(out_names, out_types)]
        node = helper.make_node(op, names, out_names, **attrs)
        g = helper.make_graph([node], f"{op}_case", graph_in, outs, inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], ir_version=9)
        name = f"ops_{op.lower()}_{self.count:04d}{('_' + tag) if tag else ''}"
        self.count += 1
        d = self.out / name
        (d / "test_data_set_0").mkdir(parents=True)
        onnx.save(m, d / "model.onnx")
        for k, (_, a) in enumerate(feeds):
            (d / "test_data_set_0" / f"input_{k}.pb").write_bytes(numpy_helper.from_array(a).SerializeToString())
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
            res = sess.run(out_names, dict(feeds))
        except Exception as e:  # ORT refuses: record it
            (d / "expect_error").write_text(str(e).splitlines()[0][:300] + "\n", encoding="utf-8")
            self.errors += 1
            self.manifest.append({"case": name, "op": op, "opset": opset, "error": True})
            return
        for j, r in enumerate(res):
            (d / "test_data_set_0" / f"output_{j}.pb").write_bytes(numpy_helper.from_array(np.asarray(r)).SerializeToString())
        self.manifest.append({"case": name, "op": op, "opset": opset, "error": False})

    # -- helpers
    def floats(self, shape, special=False, dtype=np.float32):
        a = self.rng.standard_normal(shape).astype(dtype)
        if special and a.size > 6:
            f = a.reshape(-1)
            f[self.rng.integers(f.size)] = np.nan
            f[self.rng.integers(f.size)] = np.inf
            f[self.rng.integers(f.size)] = -np.inf
            f[self.rng.integers(f.size)] = -0.0
        return a

    def ints(self, shape, lo=-50, hi=50, dtype=np.int64):
        return self.rng.integers(lo, hi, size=shape).astype(dtype)

    def dims(self, rank, choices=(1, 2, 3, 5, 8, 31, 33)):
        return [int(self.rng.choice(choices)) for _ in range(rank)]


def gen(c: Cases):
    r = c.rng
    dtypes = [np.float32, np.int64, np.int8, np.uint8, np.float16, np.float64, np.int32, np.bool_]

    def data(shape, dt):
        if dt == np.bool_:
            return r.integers(0, 2, size=shape).astype(np.bool_)
        if np.issubdtype(dt, np.floating):
            return c.floats(shape, special=True, dtype=dt)
        return c.ints(shape, -100, 100, dt)

    # Transpose
    for i in range(80):
        rank = int(r.integers(1, 6))
        shape = c.dims(rank)
        while int(np.prod(shape)) > 20000:
            shape = c.dims(rank)
        dt = dtypes[i % len(dtypes)]
        perm = [int(p) for p in r.permutation(rank)]
        c.add("Transpose", 13, [data(shape, dt)], {"perm": perm} if i % 5 else {}, consts=[0] if i % 2 else ())
    for shape in ([0, 3], [3, 0, 2], [1, 1, 1], []):
        c.add("Transpose", 13, [data(shape, np.float32)], {"perm": list(reversed(range(len(shape))))} if shape else {})

    # Slice, opset 13 (inputs) and 1 (attributes)
    for i in range(160):
        rank = int(r.integers(1, 5))
        shape = [int(x) for x in r.choice([0, 1, 2, 5, 9], size=rank)]
        if i % 4:
            shape = [max(s, 1) for s in shape]
        n = int(r.integers(1, rank + 1))
        axes = [int(a) for a in r.permutation(rank)[:n]]
        neg = i % 2 == 0
        axes_arg = [a - rank if neg else a for a in axes]
        starts = [int(r.choice(EDGE_INTS)) for _ in range(n)]
        ends = [int(r.choice(EDGE_INTS)) for _ in range(n)]
        with_steps = i % 3 != 0
        steps = [int(r.choice([1, 1, 2, 3, 7, -1, -1, -2, -4, 2**63 - 1, -(2**63) + 1])) for _ in range(n)] if with_steps else None
        ins = [data(shape, np.float32), np.array(starts, I64), np.array(ends, I64), np.array(axes_arg, I64)]
        ins.append(np.array(steps, I64) if with_steps else None)
        c.add("Slice", 13, ins, consts=[1, 2, 3, 4] if i % 2 else [], tag="in")
        if not with_steps:
            c.add("Slice", 1, [data(shape, np.float32)],
                  {"starts": [max(min(s, 2**31 - 1), -2**31) for s in starts], "ends": [max(min(s, 2**31 - 1), -2**31) for s in ends],
                   "axes": axes_arg}, tag="attr")

    # Concat / Split
    for i in range(60):
        rank = int(r.integers(1, 5))
        axis = int(r.integers(rank))
        base = [int(x) for x in r.integers(1, 5, size=rank)]
        parts = int(r.integers(1, 5))
        sizes = [int(x) for x in r.integers(0, 5, size=parts)]
        if sum(sizes) == 0:
            sizes[0] = 1
        arrs = []
        for s in sizes:
            sh = list(base)
            sh[axis] = s
            arrs.append(data(sh, np.float32))
        c.add("Concat", 13, arrs, {"axis": axis - rank if i % 2 else axis}, consts=[0] if i % 3 == 0 else ())
        whole = np.concatenate(arrs, axis=axis)
        c.add("Split", 13, [whole, np.array(sizes, I64)], {"axis": axis}, out_types=[np.float32] * parts, n_out=parts, consts=[1], tag="sizes")
        c.add("Split", 11, [whole], {"axis": axis, "split": sizes}, out_types=[np.float32] * parts, n_out=parts, tag="attr")
    for dim, parts in itertools.product([6, 7, 1, 10], [1, 2, 3, 4]):
        c.add("Split", 18, [data([dim, 2], np.float32)], {"axis": 0, "num_outputs": parts}, out_types=[np.float32] * parts, n_out=parts, tag="num")
        if dim % parts == 0:
            c.add("Split", 13, [data([dim, 2], np.float32)], {"axis": 0}, out_types=[np.float32] * parts, n_out=parts, tag="equal")

    # Squeeze / Unsqueeze / Flatten
    for shape, axes in [([2, 1, 3, 1], [1]), ([2, 1, 3, 1], [-1, 1]), ([1, 1], [0, 1]), ([1], [0]), ([2, 1, 3], [-2])]:
        c.add("Squeeze", 13, [data(shape, np.float32), np.array(axes, I64)], consts=[1])
        c.add("Squeeze", 13, [data(shape, np.float32), np.array(axes, I64)])
        c.add("Squeeze", 11, [data(shape, np.float32)], {"axes": axes})
    c.add("Squeeze", 13, [data([1, 2, 1, 3], np.float32)])
    for shape, axes in [([2, 3], [0]), ([2, 3], [2]), ([2, 3], [-1]), ([2, 3], [0, 3]), ([3], [1, 0]), ([], [0]), ([2, 3], [-3, -1]), ([2, 3], [3, 1])]:
        c.add("Unsqueeze", 13, [data(shape, np.float32), np.array(axes, I64)], consts=[1])
        c.add("Unsqueeze", 13, [data(shape, np.float32), np.array(axes, I64)])
        c.add("Unsqueeze", 11, [data(shape, np.float32)], {"axes": axes})
    for axis in range(-4, 5):
        c.add("Flatten", 13, [data([2, 3, 4], np.float32)], {"axis": axis}) if -3 <= axis <= 3 else None
    c.add("Flatten", 13, [data([0, 3], np.float32)], {"axis": 1})
    c.add("Flatten", 13, [data([2, 3, 4], np.float32)])

    # Expand / Tile
    for ind, to in [([3, 1], [2, 1, 4]), ([1], [5]), ([4], [3, 4]), ([2, 3], [1, 3]), ([1, 3], [2, 1]), ([2, 1, 3], [2, 4, 3]), ([], [3]), ([0, 1], [0, 4]),
                    ([1], [1]), ([5, 1, 1], [5, 3, 4]), ([2, 3], [2, 3])]:
        for const in (True, False):
            c.add("Expand", 13, [data(ind, np.float32), np.array(to, I64)], consts=[1] if const else ())
    c.add("Expand", 13, [data([3], np.int64), np.array([2, 3], I64)], consts=[1])
    for i in range(40):
        rank = int(r.integers(1, 4))
        shape = [int(x) for x in r.integers(0, 4, size=rank)]
        reps = [int(x) for x in r.integers(0, 4, size=rank)]
        c.add("Tile", 13, [data(shape, np.float32), np.array(reps, I64)], consts=[1] if i % 2 else ())

    # Gather family
    for i in range(80):
        rank = int(r.integers(1, 4))
        shape = [int(x) for x in r.integers(1, 5, size=rank)]
        axis = int(r.integers(rank))
        irank = int(r.integers(0, 3))
        ishape = [int(x) for x in r.integers(0, 4, size=irank)]
        idx = r.integers(-shape[axis], shape[axis], size=ishape).astype(np.int64 if i % 2 else np.int32)
        dt = dtypes[i % len(dtypes)]
        c.add("Gather", 13, [data(shape, dt), idx], {"axis": axis - rank if i % 3 == 0 else axis}, consts=[1] if i % 5 == 0 else ())
    c.add("Gather", 13, [data([3], np.float32), np.array([3], I64)])          # out of range: ORT's error
    c.add("Gather", 13, [data([3], np.float32), np.array([-4], I64)])
    for i in range(50):
        rank = int(r.integers(1, 4))
        shape = [int(x) for x in r.integers(1, 5, size=rank)]
        axis = int(r.integers(rank))
        ishape = [int(rng_choice(r, 1, s + 1)) if k != axis else int(r.integers(0, 6)) for k, s in enumerate(shape)]
        idx = r.integers(-shape[axis], shape[axis], size=ishape).astype(np.int64)
        c.add("GatherElements", 13, [data(shape, np.float32), idx], {"axis": axis})
    for i in range(60):
        batch = int(r.integers(0, 3))
        rank = batch + int(r.integers(1, 4))
        shape = [int(x) for x in r.integers(1, 4, size=rank)]
        k = int(r.integers(1, rank - batch + 1))
        lead = [int(x) for x in r.integers(1, 4, size=int(r.integers(0, 3)))]
        ishape = shape[:batch] + lead + [k]
        idx = np.zeros(ishape, np.int64)
        for j in range(k):
            idx[..., j] = r.integers(-shape[batch + j], shape[batch + j], size=ishape[:-1])
        c.add("GatherND", 13, [data(shape, np.float32), idx], {"batch_dims": batch})
    # ScatterND with unique indices (so order does not matter) and with duplicates
    for red in ["none", "add", "mul", "max", "min"]:
        for dt in (np.float32, np.int64):
            base = data([4, 3], dt)
            idx = np.array([[0], [2], [1]], np.int64)
            upd = data([3, 3], dt)
            c.add("ScatterND", 18, [base, idx, upd], {"reduction": red}, tag=red)
            idx2 = np.array([[1], [1], [3]], np.int64)
            c.add("ScatterND", 18, [base, idx2, upd], {"reduction": red}, tag=red + "_dup")
    c.add("ScatterND", 16, [data([3, 4], np.float32), np.array([[0, 1], [2, 3], [-1, -1]], np.int64), data([3], np.float32)])
    c.add("ScatterND", 16, [data([3], np.float32), np.array([[5]], np.int64), data([1], np.float32)])

    # Pad
    for i in range(120):
        rank = int(r.integers(1, 4))
        mode = ["constant", "constant", "reflect", "edge", "wrap"][i % 5]
        shape = [int(x) for x in r.integers(2 if mode == "reflect" else 1, 6, size=rank)]
        if mode == "reflect":  # ONNX Runtime only accepts pads below the extent here; larger ones are separate cases below
            before = [int(r.integers(0, s)) for s in shape]
            after = [int(r.integers(0, s)) for s in shape]
        else:
            lo = -min(shape) if mode == "constant" else 0
            before = [int(x) for x in r.integers(lo, 5, size=rank)]
            after = [int(x) for x in r.integers(lo, 5, size=rank)]
        opset = 19 if mode == "wrap" else 18
        ins = [data(shape, np.float32), np.array(before + after, I64), np.array([float(r.integers(-9, 9))], np.float32) if mode == "constant" else None]
        c.add("Pad", opset, ins, {"mode": mode}, consts=[1, 2] if i % 2 else (), tag=mode)
    c.add("Pad", 18, [data([2, 3], np.float32), np.array([1, 2], I64), None, np.array([-1], I64)], {"mode": "edge"}, consts=[1, 3], tag="axes")
    c.add("Pad", 18, [data([2, 3], np.float32), np.array([1, 0, 2, 3], I64), None, np.array([1, 0], I64)], consts=[1, 3], tag="axes")
    c.add("Pad", 11, [data([4], np.int64), np.array([2, 2], I64), np.array([7], np.int64)], consts=[1, 2], tag="int")
    c.add("Pad", 2, [data([2, 2], np.float32)], {"pads": [1, 0, 0, 1], "value": 3.5}, tag="old")
    c.add("Pad", 18, [data([1, 4], np.float32), np.array([0, 3, 0, 3], I64)], {"mode": "reflect"}, tag="reflect_big")
    c.add("Pad", 18, [data([3], np.float32), np.array([5, 5], I64)], {"mode": "reflect"}, tag="reflect_big")
    c.add("Pad", 19, [data([3], np.float32), np.array([5, 7], I64)], {"mode": "wrap"}, tag="wrap_big")
    c.add("Pad", 18, [data([1], np.float32), np.array([2, 2], I64)], {"mode": "reflect"}, tag="reflect_one")
    c.add("Pad", 18, [data([2, 0], np.float32), np.array([1, 1, 1, 1], I64)], {"mode": "constant"}, tag="empty")

    # TopK
    for i in range(80):
        rank = int(r.integers(1, 4))
        shape = [int(x) for x in r.integers(1, 4, size=rank)]
        axis = int(r.integers(rank))
        shape[axis] = int(r.choice([1, 5, 17, 40, 64]))
        k = int(r.integers(0, shape[axis] + 1))
        x = r.integers(-4, 5, size=shape).astype(np.float32)
        if i % 3 == 0 and x.size > 3:
            x.reshape(-1)[r.integers(x.size)] = np.nan
            x.reshape(-1)[r.integers(x.size)] = np.inf
        if i % 5 == 0:
            x = r.integers(-4, 5, size=shape).astype(np.int64)
        c.add("TopK", 11, [x, np.array([k], I64)], {"axis": axis, "largest": int(r.integers(2))}, out_types=[x.dtype, np.int64], n_out=2,
              consts=[1] if i % 2 else (), tag="nan" if np.isnan(x).any() else "")

    # OneHot
    for axis in (-1, 0, 1, -2):
        for idt in (np.int64, np.float32, np.int32):  # ONNX Runtime has kernels only where indices, depth and values share a type
            idx = np.array([[1, 9], [-1, 2]]).astype(idt)
            c.add("OneHot", 11, [idx, np.array(4, idt), np.array([-1, 7], idt)], {"axis": axis}, out_types=[idt])
    c.add("OneHot", 9, [np.array([0, 2], I64), np.array([3], I64), np.array([5, 6], I64)], out_types=[np.int64], consts=[1])
    c.add("OneHot", 11, [np.array([0, 2], I64), np.array([3], I64), np.array([5.0, 6.0], np.float32)], out_types=[np.float32], tag="mixed")
    c.add("OneHot", 11, [np.array([], I64), np.array(3, I64), np.array([0.0, 1.0], np.float32)], out_types=[np.float32])
    c.add("OneHot", 11, [np.array(2, I64), np.array(3, I64), np.array([0.0, 1.0], np.float32)], out_types=[np.float32], tag="scalar_indices")

    # NonZero
    for shape in ([2, 3], [5], [0, 3], [2, 2, 2], [1]):
        x = r.integers(-1, 2, size=shape).astype(np.float32)
        if x.size > 2:
            x.reshape(-1)[0] = np.nan
            x.reshape(-1)[1] = -0.0
        c.add("NonZero", 13, [x], out_types=[np.int64])
    c.add("NonZero", 13, [np.array(3.0, np.float32)], out_types=[np.int64], tag="scalar")
    c.add("NonZero", 13, [np.array(0.0, np.float32)], out_types=[np.int64], tag="scalar_zero")
    c.add("NonZero", 13, [np.array([True, False, True])], out_types=[np.int64], tag="bool")
    c.add("NonZero", 13, [np.array([0, 3, 0, 2], np.int64)], out_types=[np.int64], tag="i64")

    # Trilu
    for upper in (0, 1):
        for k in (-5, -1, 0, 1, 2, 9):
            c.add("Trilu", 14, [data([2, 3, 4], np.float32), np.array(k, I64)], {"upper": upper}, consts=[1] if k % 2 else ())
    c.add("Trilu", 14, [data([4, 4], np.int64)], {"upper": 1}, tag="no_k")
    c.add("Trilu", 14, [data([0, 3], np.float32), np.array(0, I64)], tag="empty")

    # Range
    for dt, (s, l, d) in itertools.product([np.float32, np.float64, np.int64, np.int32, np.int16],
                                           [(0, 10, 3), (10, 0, -3), (1, 0, 1), (0, 1, 0.1), (5, 5, 1), (0, 7, 2), (-3, 3, 1), (0.5, 3.3, 0.4)]):
        if np.issubdtype(dt, np.integer) and (float(s) != int(s) or float(d) != int(d) or float(l) != int(l)):
            continue
        c.add("Range", 11, [np.array(s, dt), np.array(l, dt), np.array(d, dt)], out_types=[dt], consts=[0, 1, 2] if s % 2 == 0 else ())
    c.add("Range", 11, [np.array(0, np.int64), np.array(5, np.int64), np.array(0, np.int64)], out_types=[np.int64], tag="zero_delta")
    c.add("Range", 11, [np.array(0, np.float32), np.array(np.inf, np.float32), np.array(1, np.float32)], out_types=[np.float32], tag="inf")
    c.add("Range", 11, [np.array(0, np.float32), np.array(1, np.float32), np.array(1 / 3, np.float32)], out_types=[np.float32], tag="third")
    c.add("Range", 11, [np.array(0, np.float32), np.array(0.3, np.float32), np.array(0.1, np.float32)], out_types=[np.float32], tag="tenth")
    c.add("Range", 11, [np.array(0, np.float32), np.array(3.0000002, np.float32), np.array(1, np.float32)], out_types=[np.float32], tag="ulp")

    # ConstantOfShape
    for shape in ([2, 3], [0, 2], [], [1], [4, 1, 2]):
        c.add("ConstantOfShape", 9, [np.array(shape, I64)], {"value": numpy_helper.from_array(np.array([7], np.int64), "v")}, out_types=[np.int64])
        c.add("ConstantOfShape", 9, [np.array(shape, I64)], out_types=[np.float32], consts=[0], tag="default")
    c.add("ConstantOfShape", 9, [np.array([3], I64)], {"value": numpy_helper.from_array(np.array([True]), "v")}, out_types=[np.bool_])
    c.add("ConstantOfShape", 9, [np.array([2], I64)], {"value": numpy_helper.from_array(np.array([-0.5], np.float16), "v")}, out_types=[np.float16])
    c.add("ConstantOfShape", 9, [np.array([-1], I64)], out_types=[np.float32], tag="negative")

    # Shape / Size / Dropout
    x = data([2, 1, 3, 4], np.float32)
    for s, e in [(None, None), (1, None), (None, -1), (1, -1), (-2, None), (3, 1), (-10, 10), (2, 2)]:
        at = {}
        if s is not None:
            at["start"] = s
        if e is not None:
            at["end"] = e
        c.add("Shape", 15, [x], at, out_types=[np.int64])
    c.add("Shape", 13, [x], out_types=[np.int64])
    c.add("Shape", 13, [np.array(3.0, np.float32)], out_types=[np.int64], tag="scalar")
    c.add("Size", 13, [x], out_types=[np.int64])
    c.add("Size", 13, [data([0, 4], np.float32)], out_types=[np.int64], tag="empty")
    c.add("Dropout", 13, [x], out_types=[np.float32])
    c.add("Dropout", 13, [x, np.array(0.5, np.float32), np.array(False)], out_types=[np.float32, np.bool_], n_out=2, consts=[1, 2], tag="mask")
    c.add("Dropout", 7, [x], out_types=[np.float32], tag="old")


def rng_choice(r, lo, hi):
    return r.integers(lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261029)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    c = Cases(out, a.seed)
    gen(c)
    (out / "manifest.json").write_text(
        json.dumps({"onnxruntime": ort.__version__, "onnx": onnx.__version__, "numpy": np.__version__, "seed": a.seed,
                    "cases": c.manifest}, indent=1), encoding="utf-8")
    print(f"{c.count} cases, {c.errors} that ONNX Runtime refuses; onnxruntime {ort.__version__}")


if __name__ == "__main__":
    main()
