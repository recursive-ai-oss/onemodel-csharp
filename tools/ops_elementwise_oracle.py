#!/usr/bin/env python3
"""Generates the element-wise operator oracle cases: small ONNX graphs, seeded random inputs (edge values included) and the outputs
ONNX Runtime gives for them. The C# tests (tests/OneModel.Tests/Graph/Operators/Elementwise) run the same graphs through the library
and compare. Output: testdata/ops-elementwise/cases.json.gz. Run with the oracle venv (tools/requirements-oracles.txt):

    python tools/ops_elementwise_oracle.py [out_dir]

ONNX Runtime is used as a black box through its public Python API. The case list is the table at the bottom; "tol" is the scaled
error allowed (max|ours-ref| / max|ref|, 0 = bit-equal; NaN positions and infinities must match in every case).
"""
import base64
import gzip
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

SEED = 20261029
rng = np.random.default_rng(SEED)

NP = {
    "f32": (np.float32, TensorProto.FLOAT), "f64": (np.float64, TensorProto.DOUBLE), "f16": (np.float16, TensorProto.FLOAT16),
    "i8": (np.int8, TensorProto.INT8), "i16": (np.int16, TensorProto.INT16), "i32": (np.int32, TensorProto.INT32),
    "i64": (np.int64, TensorProto.INT64), "u8": (np.uint8, TensorProto.UINT8), "u16": (np.uint16, TensorProto.UINT16),
    "u32": (np.uint32, TensorProto.UINT32), "u64": (np.uint64, TensorProto.UINT64), "bool": (np.bool_, TensorProto.BOOL),
}

SPECIAL = [np.nan, np.inf, -np.inf, -0.0, 1.0e-40, 3.0e38, -3.0e38, 0.0]

cases = []


def gen(kind, dt, shape, lo=-3.0, hi=3.0, special=True):
    """Random tensor. kind: 'f' uniform floats, 'i' integers in [lo, hi], 'b' bool, 'p' positive floats."""
    n = int(np.prod(shape)) if len(shape) else 1
    t, _ = NP[dt]
    if dt == "bool":
        a = rng.integers(0, 2, size=n).astype(np.bool_)
    elif np.issubdtype(t, np.integer):
        a = rng.integers(int(lo), int(hi) + 1, size=n).astype(t)
    else:
        a = rng.uniform(lo, hi, size=n)
        if kind == "p":
            a = np.abs(a) + 0.01
        a = a.astype(t)
        if special and n > 8:
            for v in SPECIAL:
                a[rng.integers(0, n)] = v
    return a.reshape(shape)


def add(name, op, opset, ins, attrs=None, tol=1e-6, out_dtype=None, omit=(), n_outputs=1):
    """ins: list of arrays (or None for an omitted optional input). Records the ORT result."""
    attrs = attrs or {}
    names = [f"x{i}" if a is not None else "" for i, a in enumerate(ins)]
    inputs = []
    inits = []
    for i, a in enumerate(ins):
        if a is None:
            continue
        et = {np.dtype(v[0]): v[1] for v in NP.values()}[a.dtype]
        inputs.append(helper.make_tensor_value_info(f"x{i}", et, list(a.shape)))
    node = helper.make_node(op, names, ["y"], **attrs)
    probe = None
    # output type is read back from ORT, shape too
    out = helper.make_tensor_value_info("y", _out_type(op, attrs, ins), None)
    g = helper.make_graph([node], name, inputs, [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], ir_version=9)
    try:
        s = ort.InferenceSession(m.SerializeToString(), sess_options=_opts(), providers=["CPUExecutionProvider"])
        feed = {f"x{i}": a for i, a in enumerate(ins) if a is not None}
        r = s.run(None, feed)[0]
    except Exception as e:  # unsupported by ORT: not a case
        print(f"skip {name}: {str(e).splitlines()[0][:100]}")
        return
    # rebuild the model with the concrete output type so our importer sees a declared output
    et = {np.dtype(v[0]): v[1] for v in NP.values()}.get(r.dtype)
    if et is None:
        print(f"skip {name}: output type {r.dtype}")
        return
    out = helper.make_tensor_value_info("y", et, list(r.shape))
    g = helper.make_graph([node], name, inputs, [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], ir_version=9)
    cases.append({
        "name": name, "op": op, "opset": opset, "tol": tol,
        "model": base64.b64encode(m.SerializeToString()).decode(),
        "inputs": [None if a is None else tensor(a) for a in ins],
        "output": tensor(r),
    })


def _out_type(op, attrs, ins):
    if op in ("Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual", "And", "Or", "Xor", "Not"):
        return TensorProto.BOOL
    if op == "Cast":
        return attrs["to"]
    if op == "CastLike":
        return {np.dtype(v[0]): v[1] for v in NP.values()}[ins[1].dtype]
    if op == "Where":
        return {np.dtype(v[0]): v[1] for v in NP.values()}[ins[1].dtype]
    return {np.dtype(v[0]): v[1] for v in NP.values()}[ins[0].dtype]


def _opts():
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    o.log_severity_level = 4
    return o


def tensor(a):
    a = np.asarray(a, order="C")
    dt = [k for k, v in NP.items() if np.dtype(v[0]) == a.dtype][0]
    return {"dtype": dt, "shape": list(a.shape), "data": base64.b64encode(a.astype(a.dtype.newbyteorder("<")).tobytes()).decode()}


F = "f32"
SHAPES1 = [[0], [1], [7], [3, 5], [2, 3, 4], [37], [1000]]

# ---- unary float functions -------------------------------------------------------------------------------------------
for op, opset, attrs, lo, hi, tol in [
    ("Exp", 13, {}, -20, 20, 1e-4), ("Log", 13, {}, -1, 50, 1e-4), ("Tanh", 13, {}, -6, 6, 1e-4), ("Sigmoid", 13, {}, -12, 12, 1e-4),
    ("Erf", 13, {}, -4, 4, 1e-4), ("Sqrt", 13, {}, -1, 40, 0), ("Reciprocal", 13, {}, -5, 5, 0), ("Neg", 13, {}, -5, 5, 0),
    ("Abs", 13, {}, -5, 5, 0), ("Floor", 13, {}, -5, 5, 0), ("Ceil", 13, {}, -5, 5, 0), ("Round", 11, {}, -5, 5, 0),
    ("Sign", 13, {}, -5, 5, 0), ("HardSigmoid", 6, {"alpha": 0.3, "beta": 0.4}, -6, 6, 1e-4), ("HardSigmoid", 13, {}, -6, 6, 1e-4),
    ("HardSwish", 14, {}, -6, 6, 1e-4), ("LeakyRelu", 16, {"alpha": 0.1}, -5, 5, 0), ("LeakyRelu", 6, {}, -5, 5, 0),
    ("Gelu", 20, {}, -6, 6, 1e-4), ("Gelu", 20, {"approximate": "tanh"}, -6, 6, 1e-4),
]:
    for shp in SHAPES1:
        x = gen("f", F, shp, lo, hi)
        if op == "Gelu":  # gelu(-inf) is -0 here and NaN in ONNX Runtime (-inf * 0); documented in docs/ops-elementwise.md
            x[np.isneginf(x)] = -1e30
        add(f"{op}_f32_{'x'.join(map(str, shp))}_o{opset}_{len(attrs)}", op, opset, [x], attrs, tol)
# rounding specials: halves
add("Round_halves", "Round", 11, [np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.4999, 2.5000001, -0.0, 1e10], np.float32)], tol=0)
add("Sign_zero", "Sign", 13, [np.array([0.0, -0.0, 1e-30, -1e-30, np.nan, np.inf, -np.inf], np.float32)], tol=0)

# ---- unary on other types -------------------------------------------------------------------------------------------------
for dt in ["i8", "i16", "i32", "i64"]:
    for op in ["Neg", "Abs", "Sign"]:
        add(f"{op}_{dt}", op, 13, [gen("i", dt, [3, 7], -100, 100)], tol=0)
for dt in ["u8", "u16", "u32", "u64"]:
    for op in ["Abs", "Sign"]:
        add(f"{op}_{dt}", op, 13, [gen("i", dt, [3, 7], 0, 100)], tol=0)
for op, lo, hi in [("Neg", -5, 5), ("Abs", -5, 5), ("Sign", -5, 5), ("Floor", -5, 5), ("Ceil", -5, 5), ("Round", -5, 5), ("Sqrt", -1, 40), ("Reciprocal", -5, 5)]:
    add(f"{op}_f64", op, 13 if op != "Round" else 11, [gen("f", "f64", [4, 9], lo, hi)], tol=0)
for op in ["Neg", "Abs", "Sign", "Floor", "Ceil", "Round", "Sqrt", "Reciprocal"]:
    add(f"{op}_f16", op, 13 if op != "Round" else 11, [gen("f", "f16", [4, 9], 0.5, 8, special=False)], tol=0)

# ---- binary arithmetic ----------------------------------------------------------------------------------------------------
PAIRS = [([3, 4], [3, 4]), ([3, 4], [4]), ([3, 1], [1, 4]), ([2, 1, 4], [3, 1]), ([], [5]), ([5], []), ([0, 3], [1, 3]), ([1], [1]),
         ([2, 3, 4, 5], [4, 1]), ([1, 1000], [1000]), ([37], [37]), ([2, 3, 4], [2, 3, 4])]
for op in ["Sub", "Div", "Max", "Min", "Sum", "Mean", "Add", "Mul"]:
    for dt in ["f32"]:
        for i, (sa, sb) in enumerate(PAIRS):
            add(f"{op}_{dt}_pair{i}", op, 13, [gen("f", dt, sa, -4, 4), gen("f", dt, sb, -4, 4)], tol=0 if op not in ("Mean",) else 1e-6)
    for dt in (["i32", "i64", "i8", "u8", "i16", "u32", "u64", "i32"] if op in ("Sub", "Div", "Add", "Mul") else ["i32", "i64", "i8", "u8", "u64", "f64"]):
        lo = 1 if op == "Div" and dt.startswith("u") else -50
        for i, (sa, sb) in enumerate(PAIRS[:5]):
            b = gen("i", dt, sb, lo, 60) if dt not in ("f64",) else gen("f", dt, sb, -4, 4)
            if op == "Div" and dt != "f64":
                b = np.where(b == 0, 1, b).astype(b.dtype)
            a = gen("i", dt, sa, lo, 60) if dt not in ("f64",) else gen("f", dt, sa, -4, 4)
            add(f"{op}_{dt}_pair{i}", op, 13, [a, b], tol=0)
    for dt in ["f64"]:
        add(f"{op}_{dt}_bc", op, 13, [gen("f", dt, [3, 1, 4], -4, 4), gen("f", dt, [5, 1], -4, 4)], tol=0 if op != "Mean" else 1e-12)
for op in ["Max", "Min", "Sum", "Mean"]:
    for opset in (8, 13):
        add(f"{op}_three_o{opset}", op, opset, [gen("f", F, [3, 4], -3, 3), gen("f", F, [4], -3, 3), gen("f", F, [3, 1], -3, 3)], tol=0 if op != "Mean" else 1e-6)
    add(f"{op}_one", op, 13, [gen("f", F, [2, 5], -3, 3)], tol=0)
    add(f"{op}_five_grow", op, 13, [gen("f", F, [1], -3, 3), gen("f", F, [1, 4], -3, 3), gen("f", F, [3, 1], -3, 3), gen("f", F, [2, 1, 1], -3, 3), gen("f", F, [4], -3, 3)], tol=0 if op != "Mean" else 1e-6)
    add(f"{op}_o6_same", op, 6, [gen("f", F, [3, 4], -3, 3), gen("f", F, [3, 4], -3, 3)], tol=0 if op != "Mean" else 1e-6)
for dt in ["i32", "i64", "u8", "i8"]:
    for op in ["Max", "Min"]:
        add(f"{op}_{dt}_three", op, 13, [gen("i", dt, [3, 4], 0, 100), gen("i", dt, [4], 0, 100), gen("i", dt, [3, 1], 0, 100)], tol=0)

# Div edge: zero, inf, nan
add("Div_f32_edge", "Div", 13, [np.array([1, -1, 0, 0, np.inf, np.nan, 1, 5], np.float32), np.array([0, 0, 0, np.inf, np.inf, 1, np.nan, -2], np.float32)], tol=0)
add("Max_f32_edge", "Max", 13, [np.array([1, np.nan, 0, -0.0, np.inf, 2, np.nan], np.float32), np.array([np.nan, 1, -0.0, 0, 1, 2, np.nan], np.float32)], tol=0)
add("Min_f32_edge", "Min", 13, [np.array([1, np.nan, 0, -0.0, np.inf, 2, np.nan], np.float32), np.array([np.nan, 1, -0.0, 0, 1, 2, np.nan], np.float32)], tol=0)

# ---- Mod -------------------------------------------------------------------------------------------------------------------
for dt in ["i32", "i64", "i8", "u8", "u32"]:
    lo = 0 if dt.startswith("u") else -40
    a = gen("i", dt, [4, 6], lo, 40)
    b = gen("i", dt, [6], lo, 20)
    b = np.where(b == 0, 3, b).astype(b.dtype)
    add(f"Mod_{dt}_floor", "Mod", 13, [a, b], {"fmod": 0}, tol=0)
    add(f"Mod_{dt}_fmod", "Mod", 13, [a, b], {"fmod": 1}, tol=0)
add("Mod_f32_fmod", "Mod", 13, [gen("f", F, [4, 6], -9, 9, special=False), gen("f", F, [6], -4, 4, special=False)], {"fmod": 1}, tol=0)
add("Mod_f32_fmod_edge", "Mod", 13, [np.array([5.5, -5.5, 5.5, -5.5, np.inf, np.nan, 3, 0], np.float32), np.array([2, 2, -2, -2, 2, 2, np.inf, 0], np.float32)], {"fmod": 1}, tol=0)
add("Mod_f64_fmod", "Mod", 13, [gen("f", "f64", [3, 5], -9, 9, special=False), gen("f", "f64", [5], -4, 4, special=False)], {"fmod": 1}, tol=0)

# ---- Pow -------------------------------------------------------------------------------------------------------------------
for opset in (7, 12, 13, 15):
    add(f"Pow_f32_f32_o{opset}", "Pow", opset, [gen("p", F, [3, 5], 0.1, 3, special=False), gen("f", F, [5], -3, 3, special=False)], tol=1e-6)
add("Pow_f32_two", "Pow", 13, [gen("f", F, [4, 6], -3, 3), np.array(2.0, np.float32)], tol=1e-6)
add("Pow_f32_half", "Pow", 13, [gen("p", F, [4, 6], 0.1, 3, special=False), np.array(0.5, np.float32)], tol=1e-6)
add("Pow_f32_int", "Pow", 13, [gen("f", F, [4, 6], -3, 3, special=False), np.array([-2, -1, 0, 1, 2, 3], np.int32)], tol=1e-6)
add("Pow_f32_i64", "Pow", 15, [gen("f", F, [4, 6], -3, 3, special=False), np.array([-2, -1, 0, 1, 2, 3], np.int64)], tol=1e-6)
add("Pow_f64_f64", "Pow", 15, [gen("p", "f64", [4, 6], 0.1, 3, special=False), gen("f", "f64", [6], -3, 3, special=False)], tol=1e-12)
add("Pow_f64_f32", "Pow", 15, [gen("p", "f64", [4, 6], 0.1, 3, special=False), gen("f", F, [6], -3, 3, special=False)], tol=1e-12)
add("Pow_i32_i32", "Pow", 13, [gen("i", "i32", [4, 6], -3, 5), gen("i", "i32", [6], 0, 4)], tol=0)
add("Pow_i64_i64", "Pow", 13, [gen("i", "i64", [4, 6], -3, 5), gen("i", "i64", [6], 0, 4)], tol=0)
add("Pow_i32_f32", "Pow", 15, [gen("i", "i32", [4, 6], 1, 5), np.array(2.0, np.float32)], tol=0)
add("Pow_edge", "Pow", 13, [np.array([0, 0, -8, 2, np.inf, np.nan, 1, -2], np.float32), np.array([0, -1, 1 / 3, np.inf, 0, 0, np.nan, 2], np.float32)], tol=1e-6)

# ---- comparisons and logic ------------------------------------------------------------------------------------------------
for op, opset in [("Equal", 13), ("Less", 13), ("Greater", 13), ("LessOrEqual", 16), ("GreaterOrEqual", 16)]:
    for i, (sa, sb) in enumerate(PAIRS[:7]):
        add(f"{op}_f32_pair{i}", op, opset, [gen("i", "i32", sa, -3, 3).astype(np.float32), gen("i", "i32", sb, -3, 3).astype(np.float32)], tol=0)
    add(f"{op}_f32_nan", op, opset, [np.array([1, np.nan, np.nan, np.inf, -np.inf, 0, -0.0, 2], np.float32), np.array([np.nan, 1, np.nan, np.inf, np.inf, -0.0, 0, 2], np.float32)], tol=0)
    for dt in ["i32", "i64", "i8", "u8", "u32", "u64", "f64", "i16", "u16"]:
        add(f"{op}_{dt}", op, opset, [gen("i", dt, [3, 5], 0, 4), gen("i", dt, [5], 0, 4)], tol=0)
for dt in ["bool"]:
    add("Equal_bool", "Equal", 13, [gen("b", dt, [3, 5]), gen("b", dt, [5])], tol=0)
for op in ["And", "Or", "Xor"]:
    for i, (sa, sb) in enumerate(PAIRS[:7]):
        add(f"{op}_pair{i}", op, 13, [gen("b", "bool", sa), gen("b", "bool", sb)], tol=0)
add("Not_bool", "Not", 1, [gen("b", "bool", [3, 7])], tol=0)
add("Not_bool_scalar", "Not", 13, [np.array(True)], tol=0)

# ---- Where -----------------------------------------------------------------------------------------------------------------
for dt in ["f32", "i64", "bool", "u8", "i32", "f64"]:
    for i, (sc, sx, sy) in enumerate([([3, 4], [3, 4], [3, 4]), ([4], [3, 4], [1]), ([3, 1], [1, 4], [3, 4]), ([], [2, 3], [2, 3]), ([2, 1, 4], [3, 1], [4]), ([0, 3], [1, 3], [3])]):
        add(f"Where_{dt}_{i}", "Where", 16, [gen("b", "bool", sc), gen("f" if dt in ("f32", "f64") else "i", dt, sx, 0, 100), gen("f" if dt in ("f32", "f64") else "i", dt, sy, 0, 100)], tol=0)

# ---- Clip ------------------------------------------------------------------------------------------------------------------
x = gen("f", F, [5, 9], -4, 4)
add("Clip_o6_both", "Clip", 6, [x], {"min": -1.0, "max": 2.0}, tol=0)
add("Clip_o6_min", "Clip", 6, [x], {"min": -1.0}, tol=0)
add("Clip_o6_max", "Clip", 6, [x], {"max": 0.5}, tol=0)
add("Clip_o6_none", "Clip", 6, [x], {}, tol=0)
add("Clip_o11_both", "Clip", 11, [x, np.array(-1.0, np.float32), np.array(2.0, np.float32)], tol=0)
add("Clip_o11_min", "Clip", 11, [x, np.array(-1.0, np.float32)], tol=0)
add("Clip_o13_max", "Clip", 13, [x, None, np.array(1.5, np.float32)], tol=0)
add("Clip_o13_none", "Clip", 13, [x], tol=0)
add("Clip_o13_inverted", "Clip", 13, [x, np.array(2.0, np.float32), np.array(-1.0, np.float32)], tol=0)
for dt in ["i32", "i64", "i8", "u8", "f64"]:
    t = NP[dt][0]
    xi = gen("i", dt, [4, 7], 0, 100) if dt != "f64" else gen("f", dt, [4, 7], -4, 4)
    add(f"Clip_{dt}", "Clip", 13, [xi, np.array(10, t), np.array(50, t)], tol=0)

# ---- Cast ------------------------------------------------------------------------------------------------------------------
TO = {"f32": 1, "u8": 2, "i8": 3, "u16": 4, "i16": 5, "i32": 6, "i64": 7, "bool": 9, "f16": 10, "f64": 11, "u32": 12, "u64": 13}
def cast_case(src, dst, x):
    add(f"Cast_{src}_to_{dst}", "Cast", 13, [x], {"to": TO[dst]}, tol=0)
fl = gen("f", F, [4, 9], -100, 100, special=False)
fl[0, 0] = 0.5; fl[0, 1] = -0.5; fl[0, 2] = 1.5; fl[0, 3] = 2.9; fl[0, 4] = -2.9; fl[0, 5] = 0
for dst in ["f64", "f16", "i32", "i64", "i8", "i16", "bool"]:
    cast_case("f32", dst, fl)
fpos = np.abs(fl)
for dst in ["u8", "u16", "u32", "u64"]:
    cast_case("f32", dst, fpos)
cast_case("f32", "bool", np.array([0, -0.0, np.nan, np.inf, 1e-45, 2], np.float32))
cast_case("f32", "f32", fl)
cast_case("f32", "f16", np.array([65504, 65520, 1e-8, 6e-8, 6.1e-5, 0.1, np.inf, -np.inf, np.nan, 1.00048828125, 1.0009765625], np.float32))
big = gen("i", "i32", [4, 9], -1000, 1000)
for dst in ["f32", "f64", "i64", "i8", "u8", "i16", "u16", "u32", "u64", "bool", "f16"]:
    cast_case("i32", dst, big)
cast_case("i32", "f32", np.array([16777217, -16777217, 2147483647, -2147483648, 0], np.int32))
b64 = np.array([0, 1, -1, 2**40, -(2**40), 2**62, -(2**62), 9007199254740993, 2**63 - 1, -(2**63)], np.int64)
for dst in ["f32", "f64", "i32", "u64", "u8", "bool", "i16"]:
    cast_case("i64", dst, b64)
u64 = np.array([0, 1, 2**63, 2**64 - 1, 2**40, 12345678901234567890], np.uint64)
for dst in ["f32", "f64", "i64", "u32", "bool"]:
    cast_case("u64", dst, u64)
for dst in ["f32", "i32", "i64", "u8", "f64"]:
    cast_case("bool", dst, gen("b", "bool", [3, 7]))
for dst in ["f32", "i32", "i64"]:
    cast_case("u8", dst, gen("i", "u8", [3, 7], 0, 255))
    cast_case("i8", dst, gen("i", "i8", [3, 7], -128, 127))
for dst in ["f32", "f16", "i32", "i64"]:
    cast_case("f64", dst, gen("f", "f64", [3, 7], -100, 100, special=False))
for dst in ["f32", "f64", "i32", "bool"]:
    cast_case("f16", dst, gen("f", "f16", [3, 7], -100, 100, special=False))
cast_case("f64", "f32", np.array([1e300, -1e300, 1e-300, 0.1, 16777217.0, np.nan, np.inf], np.float64))
add("Cast_f32_empty", "Cast", 13, [np.zeros([0, 3], np.float32)], {"to": 6}, tol=0)
add("Cast_scalar", "Cast", 13, [np.array(3.7, np.float32)], {"to": 7}, tol=0)
for opset in (6, 9, 19, 21):
    add(f"Cast_o{opset}", "Cast", opset, [fl], {"to": 6}, tol=0)
# CastLike
for src, dst in [("f32", "i32"), ("i32", "f32"), ("f32", "f16"), ("i64", "bool"), ("f32", "f64")]:
    like = gen("i", dst, [1], 0, 1) if dst != "bool" else np.array([True])
    like = like.astype(NP[dst][0])
    xs = gen("f", src, [3, 4], -50, 50, special=False) if src[0] == "f" else gen("i", src, [3, 4], -50, 50)
    add(f"CastLike_{src}_{dst}", "CastLike", 15, [xs, like], tol=0)

# ---- float16 arithmetic and unsigned Pow exponents (appended last so earlier cases keep their draws) ----------------------------
for op in ["Add", "Sub", "Mul", "Div", "Max", "Min"]:
    add(f"{op}_f16", op, 13, [gen("f", "f16", [3, 4], -4, 4, special=False), gen("f", "f16", [4], 0.5, 4, special=False)], tol=0)
add("Mod_f16_fmod", "Mod", 13, [gen("f", "f16", [3, 4], -9, 9, special=False), gen("f", "f16", [4], 1, 4, special=False)], {"fmod": 1}, tol=0)
for dt in ["u32", "u64"]:
    add(f"Pow_f32_{dt}", "Pow", 15, [gen("p", F, [3, 4], 0.5, 3, special=False), gen("i", dt, [4], 0, 4)], tol=1e-6)

# MobileNetV3's HardSigmoid (alpha = 1/6, beta = 0.5), over its activation range
add("HardSigmoid_mobilenet", "HardSigmoid", 17, [gen("f", F, [2, 16, 7, 7], -8, 8)], {"alpha": 0.16666667, "beta": 0.5}, tol=1e-6)

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "testdata", "ops-elementwise")
os.makedirs(out_dir, exist_ok=True)
doc = {"generator": "tools/ops_elementwise_oracle.py", "seed": SEED, "onnxruntime": ort.__version__, "onnx": onnx.__version__,
       "numpy": np.__version__, "cases": cases}
with gzip.GzipFile(os.path.join(out_dir, "cases.json.gz"), "wb", mtime=0) as f:
    f.write(json.dumps(doc, separators=(",", ":")).encode())
print(f"{len(cases)} cases")
