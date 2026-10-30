#!/usr/bin/env python3
"""Generates the oracle cases for the operators added for backend-test coverage: small ONNX graphs, seeded random inputs and the
outputs ONNX Runtime gives for them. tests/OneModel.Tests/Graph/Operators/Coverage runs the same graphs through the library and
compares. Output: testdata/ops-coverage/cases.json.gz, in the format of tools/ops_elementwise_oracle.py ("tol" is the scaled error
allowed, max|ours-ref| / max|ref|, 0 = bit-equal). Run with the oracle venv (tools/requirements-oracles.txt):

    python tools/ops_coverage_oracle.py [out_dir]

ONNX Runtime is used as a black box through its public Python API. A case ONNX Runtime refuses is skipped and named on stderr;
only the first output of each node is recorded.
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

rng = np.random.default_rng(20261030)

NP = {
    "f32": (np.float32, TensorProto.FLOAT), "f64": (np.float64, TensorProto.DOUBLE),
    "i8": (np.int8, TensorProto.INT8), "i16": (np.int16, TensorProto.INT16), "i32": (np.int32, TensorProto.INT32),
    "i64": (np.int64, TensorProto.INT64), "u8": (np.uint8, TensorProto.UINT8), "u16": (np.uint16, TensorProto.UINT16),
    "u32": (np.uint32, TensorProto.UINT32), "u64": (np.uint64, TensorProto.UINT64), "bool": (np.bool_, TensorProto.BOOL),
}
ETYPE = {np.dtype(v[0]): v[1] for v in NP.values()}
cases = []


def f(*shape, lo=-3.0, hi=3.0, dt=np.float32):
    return rng.uniform(lo, hi, size=shape).astype(dt)


def i(*shape, lo=0, hi=5, dt=np.int64):
    return rng.integers(lo, hi + 1, size=shape).astype(dt)


def opts():
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    o.log_severity_level = 4
    return o


def tensor(a):
    a = np.asarray(a, order="C")
    dt = [k for k, v in NP.items() if np.dtype(v[0]) == a.dtype][0]
    return {"dtype": dt, "shape": list(a.shape), "data": base64.b64encode(a.astype(a.dtype.newbyteorder("<")).tobytes()).decode()}


def add(name, op, opset, ins, attrs=None, tol=1e-5, n_out=1):
    attrs = attrs or {}
    names = [f"x{k}" if a is not None else "" for k, a in enumerate(ins)]
    outs = [f"y{k}" for k in range(n_out)]
    inputs = [helper.make_tensor_value_info(f"x{k}", ETYPE[a.dtype], list(a.shape)) for k, a in enumerate(ins) if a is not None]
    node = helper.make_node(op, names, outs, **attrs)

    def model(out_info):
        g = helper.make_graph([node], name, inputs, [out_info])
        return helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], ir_version=10)

    try:
        # The output type is not known until shape inference has run over a graph whose output is declared without one.
        m = onnx.shape_inference.infer_shapes(model(helper.make_tensor_value_info("y0", TensorProto.UNDEFINED, None)))
        s = ort.InferenceSession(m.SerializeToString(), sess_options=opts(), providers=["CPUExecutionProvider"])
        r = s.run(None, {f"x{k}": a for k, a in enumerate(ins) if a is not None})[0]
    except Exception as e:  # not supported by ONNX Runtime: not a case
        print(f"skip {name}: {str(e).splitlines()[0][:110]}", file=sys.stderr)
        return
    if r.dtype not in ETYPE:
        print(f"skip {name}: output type {r.dtype}", file=sys.stderr)
        return
    node.output[:] = outs[:1] if n_out == 1 else outs
    m = model(helper.make_tensor_value_info("y0", ETYPE[r.dtype], list(r.shape)))
    cases.append({"name": name, "op": op, "opset": opset, "tol": tol, "model": base64.b64encode(m.SerializeToString()).decode(),
                  "inputs": [None if a is None else tensor(a) for a in ins], "output": tensor(r)})


# ---- activations --------------------------------------------------------------------------------------------------------
for shp in [(1,), (7,), (3, 5), (2, 3, 4), (37,)]:
    x = f(*shp, lo=-6, hi=6)
    tag = "x".join(map(str, shp))
    for op, opset, attrs in [("Elu", 6, {"alpha": 0.7}), ("Elu", 6, {}), ("Selu", 6, {}), ("Selu", 6, {"alpha": 1.2, "gamma": 0.9}),
                             ("Celu", 12, {"alpha": 1.3}), ("Softplus", 1, {}), ("Softsign", 1, {}), ("Mish", 18, {}),
                             ("ThresholdedRelu", 10, {"alpha": 0.5}), ("Shrink", 9, {"lambd": 0.7, "bias": 0.2}), ("Swish", 24, {"alpha": 1.5})]:
        add(f"{op}_{tag}_{len(attrs)}", op, opset, [x], attrs)
add("Softplus_large", "Softplus", 1, [np.array([-80, -20, 0, 20, 80, 100], np.float32)])
add("PRelu_vec", "PRelu", 9, [f(2, 3, 4), f(4)])
add("PRelu_chan", "PRelu", 9, [f(2, 3, 4), f(3, 1)])
add("PRelu_scalar", "PRelu", 9, [f(5, 6), f(1)])
add("PRelu_full", "PRelu", 9, [f(2, 3), f(2, 3)])
for op, lo, hi in [("Sin", -8, 8), ("Cos", -8, 8), ("Tan", -1.4, 1.4), ("Asin", -1, 1), ("Acos", -1, 1), ("Atan", -8, 8),
                   ("Sinh", -4, 4), ("Cosh", -4, 4), ("Asinh", -8, 8), ("Acosh", 1, 9), ("Atanh", -0.95, 0.95)]:
    add(f"{op}_f32", op, 22, [f(3, 9, lo=lo, hi=hi)])
    add(f"{op}_f64", op, 22, [f(5, lo=lo, hi=hi, dt=np.float64)], tol=1e-12)

# ---- bitwise, shifts, IsInf -------------------------------------------------------------------------------------------
for dt in [np.int32, np.uint8, np.int64, np.uint16]:
    lim = 100 if dt != np.uint8 else 250
    a, b = i(3, 4, lo=0, hi=lim, dt=dt), i(4, lo=0, hi=lim, dt=dt)
    for op in ["BitwiseAnd", "BitwiseOr", "BitwiseXor"]:
        add(f"{op}_{np.dtype(dt).name}", op, 18, [a, b], tol=0)
    add(f"BitwiseNot_{np.dtype(dt).name}", "BitwiseNot", 18, [a], tol=0)
for dt in [np.uint8, np.uint32, np.uint64]:
    bits = np.dtype(dt).itemsize * 8
    for d in ("LEFT", "RIGHT"):
        add(f"BitShift_{d}_{np.dtype(dt).name}", "BitShift", 11, [i(3, 5, lo=0, hi=250, dt=dt), i(5, lo=0, hi=bits - 1, dt=dt)], {"direction": d}, tol=0)
inf = np.array([[1, np.inf, -np.inf, np.nan, 0, -2]], np.float32)
for attrs in [{}, {"detect_negative": 0}, {"detect_positive": 0}]:
    add(f"IsInf_{len(attrs)}_{list(attrs.values())}", "IsInf", 20, [inf], attrs, tol=0)

# ---- losses ----------------------------------------------------------------------------------------------------------------
for red in ["none", "sum", "mean"]:
    for shp in [(5, 4), (3, 4, 2), (2, 3, 2, 2)]:
        n, c = shp[0], shp[1]
        lp = np.log(f(*shp, lo=0.05, hi=1)).astype(np.float32)
        tgt = i(n, *shp[2:], lo=0, hi=c - 1)
        w = f(c, lo=0.1, hi=2)
        tag = f"{red}_{'x'.join(map(str, shp))}"
        add(f"NLL_{tag}", "NegativeLogLikelihoodLoss", 13, [lp, tgt], {"reduction": red})
        add(f"NLL_w_{tag}", "NegativeLogLikelihoodLoss", 13, [lp, tgt, w], {"reduction": red})
        t2 = tgt.copy()
        t2.reshape(-1)[0] = -1
        add(f"NLL_ignore_{tag}", "NegativeLogLikelihoodLoss", 13, [lp, t2, w], {"reduction": red, "ignore_index": -1})
        add(f"SCE_{tag}", "SoftmaxCrossEntropyLoss", 13, [f(*shp), tgt], {"reduction": red})
        add(f"SCE_w_{tag}", "SoftmaxCrossEntropyLoss", 13, [f(*shp), tgt, w], {"reduction": red})
        add(f"SCE_ignore_{tag}", "SoftmaxCrossEntropyLoss", 13, [f(*shp), t2, w], {"reduction": red, "ignore_index": -1})
        add(f"SCE_logp_{tag}", "SoftmaxCrossEntropyLoss", 13, [f(*shp), tgt], {"reduction": red}, n_out=2)

# ---- scatter, compress, unique -----------------------------------------------------------------------------------------
for axis in [0, 1, -1]:
    d = f(3, 4)
    idx = np.stack([rng.permutation(3 if axis == 0 else 4)[:2] for _ in range(4 if axis == 0 else 3)], axis=(1 if axis == 0 else 0)).astype(np.int64)
    upd = f(*idx.shape)
    for red in ["none", "add", "mul"]:
        add(f"ScatterElements_{axis}_{red}", "ScatterElements", 16, [d, idx, upd], {"axis": axis, "reduction": red})
    for red in ["max", "min"]:
        add(f"ScatterElements_{axis}_{red}", "ScatterElements", 18, [d, idx, upd], {"axis": axis, "reduction": red})
add("ScatterElements_dup_add", "ScatterElements", 16, [np.zeros((4,), np.float32), np.array([1, 1, 2, 1], np.int64), np.array([1, 2, 3, 4], np.float32)], {"reduction": "add"})
add("ScatterElements_negative", "ScatterElements", 13, [f(5), np.array([-1, -5], np.int64), f(2)])
add("ScatterElements_i32", "ScatterElements", 13, [i(3, 3, dt=np.int32), np.array([[0, 1, 2]], np.int64), i(1, 3, dt=np.int32)], {"axis": 0}, tol=0)
for axis in [None, 0, 1, -1]:
    attrs = {} if axis is None else {"axis": axis}
    add(f"Compress_{axis}", "Compress", 11, [f(3, 4, 2), (rng.random(4 if axis is not None else 24) > 0.5)], attrs, tol=0)
add("Compress_short", "Compress", 11, [f(3, 4), np.array([True, False, True])], {"axis": 1}, tol=0)
add("Compress_none", "Compress", 11, [f(3, 4), np.zeros(4, np.bool_)], {"axis": 1}, tol=0)
dup = np.array([[3, 1, 2], [3, 1, 2], [0, 5, 5], [1, 1, 1]], np.float32)
for sorted_ in [1, 0]:
    add(f"Unique_flat_{sorted_}", "Unique", 11, [dup], {"sorted": sorted_}, tol=0)
    add(f"Unique_axis0_{sorted_}", "Unique", 11, [dup], {"axis": 0, "sorted": sorted_}, tol=0)
    add(f"Unique_axis1_{sorted_}", "Unique", 11, [dup], {"axis": 1, "sorted": sorted_}, tol=0)
    add(f"Unique_i64_{sorted_}", "Unique", 11, [i(20, lo=0, hi=6)], {"sorted": sorted_}, tol=0)

# ---- data movement -------------------------------------------------------------------------------------------------------
for mode in ["DCR", "CRD"]:
    add(f"DepthToSpace_{mode}", "DepthToSpace", 13, [f(2, 8, 3, 2)], {"blocksize": 2, "mode": mode}, tol=0)
add("DepthToSpace_b3", "DepthToSpace", 13, [i(1, 18, 2, 2, dt=np.int32)], {"blocksize": 3}, tol=0)
add("SpaceToDepth", "SpaceToDepth", 13, [f(2, 3, 4, 6)], {"blocksize": 2}, tol=0)
for k in [0, 1, -1, 3]:
    add(f"EyeLike_{k}", "EyeLike", 9, [f(4, 5)], {"k": k}, tol=0)
add("EyeLike_i64", "EyeLike", 9, [f(3, 3)], {"dtype": TensorProto.INT64}, tol=0)
for b, t in [(1, 0), (0, 1)]:
    add(f"ReverseSequence_{b}{t}", "ReverseSequence", 10, [f(4, 4, 3), np.array([1, 4, 2, 3], np.int64)], {"batch_axis": b, "time_axis": t}, tol=0)
add("CenterCropPad_crop", "CenterCropPad", 18, [f(5, 6), np.array([3, 4], np.int64)], tol=0)
add("CenterCropPad_pad", "CenterCropPad", 18, [f(3, 4), np.array([6, 7], np.int64)], tol=0)
add("CenterCropPad_mixed_axes", "CenterCropPad", 18, [f(2, 5, 4), np.array([7, 3], np.int64)], {"axes": [1, 2]}, tol=0)
add("BitCast_f32_i32", "BitCast", 26, [f(6)], {"to": TensorProto.INT32}, tol=0)
add("TensorScatter_linear", "TensorScatter", 24, [f(2, 3, 6, 4), f(2, 3, 2, 4), np.array([1, 3], np.int64)], {"mode": "linear"}, tol=0)
add("TensorScatter_circular", "TensorScatter", 24, [f(2, 6, 4), f(2, 3, 4), np.array([5, 4], np.int64)], {"axis": 1, "mode": "circular"}, tol=0)

# ---- numeric -------------------------------------------------------------------------------------------------------------------
for ex in [0, 1]:
    for rv in [0, 1]:
        for ax in [0, 1, -1]:
            add(f"CumProd_{ex}{rv}_{ax}", "CumProd", 26, [f(3, 4, lo=0.5, hi=1.5), np.array(ax, np.int64)], {"exclusive": ex, "reverse": rv})
for p in [1, 2]:
    for ax in [-1, 0, 1]:
        add(f"LpNormalization_{p}_{ax}", "LpNormalization", 1, [f(3, 4, 5)], {"p": p, "axis": ax})
add("MeanVarianceNormalization", "MeanVarianceNormalization", 13, [f(2, 3, 4, 5)])
add("MeanVarianceNormalization_axes", "MeanVarianceNormalization", 13, [f(2, 3, 4, 5)], {"axes": [1, 3]})
for size in [1, 3, 4, 5]:
    add(f"LRN_{size}", "LRN", 13, [f(2, 6, 3, 3)], {"size": size, "alpha": 0.01, "beta": 0.6, "bias": 1.5})
add("Det_2d", "Det", 11, [f(4, 4)])
add("Det_batch", "Det", 11, [f(2, 3, 3, 3)])
add("Det_singular", "Det", 11, [np.ones((3, 3), np.float32)])
for eq, shapes in [("ij,jk->ik", [(3, 4), (4, 5)]), ("bij,bjk->bik", [(2, 3, 4), (2, 4, 5)]), ("ii->i", [(4, 4)]), ("ii", [(4, 4)]),
                   ("ij->ji", [(3, 5)]), ("ij->", [(3, 5)]), ("i,i->", [(6,), (6,)]), ("i,j->ij", [(3,), (4,)]), ("...ij,...jk->...ik", [(2, 2, 3, 4), (2, 2, 4, 3)]),
                   ("ij,jk", [(3, 4), (4, 2)]), ("bij->bj", [(2, 3, 4)]), ("abc,cd,de->abe", [(2, 3, 4), (4, 5), (5, 2)])]:
    add(f"Einsum_{eq}", "Einsum", 12, [f(*s) for s in shapes], {"equation": eq})
add("Einsum_i64", "Einsum", 12, [i(3, 4, dt=np.int64), i(4, 2, dt=np.int64)], {"equation": "ij,jk->ik"}, tol=0)

# ---- pooling and layout ---------------------------------------------------------------------------------------------------------
add("LpPool_2d", "LpPool", 18, [f(1, 2, 6, 6)], {"kernel_shape": [3, 3], "strides": [2, 2], "p": 2})
add("LpPool_1d_p3", "LpPool", 18, [f(2, 3, 9)], {"kernel_shape": [3], "p": 3})
add("LpPool_pads", "LpPool", 18, [f(1, 1, 5, 5)], {"kernel_shape": [2, 2], "pads": [1, 1, 1, 1], "p": 1})
add("LpPool_3d", "LpPool", 18, [f(1, 2, 4, 4, 4)], {"kernel_shape": [2, 2, 2], "p": 2})
x = f(1, 1, 2, 2)
idx = np.array([[[[0, 3], [8, 15]]]], np.int64)
add("MaxUnpool", "MaxUnpool", 11, [x, idx], {"kernel_shape": [2, 2], "strides": [2, 2]}, tol=0)
add("MaxUnpool_shape", "MaxUnpool", 11, [x, idx, np.array([1, 1, 5, 5], np.int64)], {"kernel_shape": [2, 2], "strides": [2, 2]}, tol=0)
add("Col2Im", "Col2Im", 18, [f(1, 4 * 2, 9), np.array([4, 4], np.int64), np.array([2, 2], np.int64)])
add("Col2Im_strides_pads", "Col2Im", 18, [f(2, 9, 8), np.array([5, 4], np.int64), np.array([3, 3], np.int64)], {"strides": [2, 1], "pads": [1, 1, 1, 1]})
add("Col2Im_dilations", "Col2Im", 18, [f(1, 4, 4), np.array([5, 5], np.int64), np.array([2, 2], np.int64)], {"dilations": [2, 2]})
add("Conv3d", "Conv", 13, [f(2, 3, 5, 6, 4), f(4, 3, 2, 3, 2), f(4)], {"kernel_shape": [2, 3, 2]})
add("Conv3d_group_stride_pad", "Conv", 13, [f(1, 4, 6, 6, 6), f(6, 2, 3, 3, 3)], {"group": 2, "strides": [2, 1, 2], "pads": [1, 0, 1, 1, 1, 0], "dilations": [1, 2, 1]})

# ---- attention ---------------------------------------------------------------------------------------------------------------------
add("Attention_4d", "Attention", 23, [f(2, 3, 4, 8), f(2, 3, 6, 8), f(2, 3, 6, 10)])
add("Attention_3d", "Attention", 23, [f(2, 4, 12), f(2, 6, 12), f(2, 6, 12)], {"q_num_heads": 3, "kv_num_heads": 3})
add("Attention_gqa", "Attention", 23, [f(2, 6, 4, 8), f(2, 2, 6, 8), f(2, 2, 6, 8)])
add("Attention_causal", "Attention", 23, [f(1, 2, 5, 4), f(1, 2, 5, 4), f(1, 2, 5, 4)], {"is_causal": 1})
add("Attention_scale_softcap", "Attention", 23, [f(1, 2, 5, 4), f(1, 2, 5, 4), f(1, 2, 5, 4)], {"scale": 0.3, "softcap": 2.0})
add("Attention_float_mask", "Attention", 23, [f(2, 2, 4, 4), f(2, 2, 6, 4), f(2, 2, 6, 4), f(4, 6)])
add("Attention_bool_mask", "Attention", 23, [f(2, 2, 4, 4), f(2, 2, 6, 4), f(2, 2, 6, 4), np.tril(np.ones((4, 6), np.bool_))])
add("Attention_past", "Attention", 23, [f(1, 2, 3, 4), f(1, 2, 2, 4), f(1, 2, 2, 4), None, f(1, 2, 4, 4), f(1, 2, 4, 4)], n_out=3)
add("Attention_qk_out", "Attention", 23, [f(1, 2, 3, 4), f(1, 2, 5, 4), f(1, 2, 5, 4)], {"qk_matmul_output_mode": 3}, n_out=1)
add("RotaryEmbedding_4d", "RotaryEmbedding", 23, [f(2, 3, 4, 8), f(6, 4), f(6, 4), i(2, 4, lo=0, hi=5)])
add("RotaryEmbedding_interleaved", "RotaryEmbedding", 23, [f(2, 3, 4, 8), f(6, 4), f(6, 4), i(2, 4, lo=0, hi=5)], {"interleaved": 1})
add("RotaryEmbedding_3d_partial", "RotaryEmbedding", 23, [f(2, 4, 24), f(2, 4, 2), f(2, 4, 2)], {"num_heads": 3, "rotary_embedding_dim": 4})

# ---- spectral, sampling, detection --------------------------------------------------------------------------------------------------
for n in [8, 10, 7]:
    for inv in [0, 1]:
        add(f"DFT_{n}_{inv}", "DFT", 20, [f(2, n, 2), None, np.array(1, np.int64)], {"inverse": inv}, tol=1e-5)
    add(f"DFT_real_{n}", "DFT", 20, [f(2, n, 1), None, np.array(1, np.int64)], tol=1e-5)
    add(f"DFT_rfft_{n}", "DFT", 20, [f(2, n, 1), None, np.array(1, np.int64)], {"onesided": 1}, tol=1e-5)
add("DFT_length", "DFT", 20, [f(2, 6, 1), np.array(9, np.int64), np.array(1, np.int64)], tol=1e-5)
add("DFT_axis2", "DFT", 20, [f(2, 3, 8, 2), None, np.array(2, np.int64)], tol=1e-5)
add("STFT", "STFT", 17, [f(2, 64, 1), np.array(8, np.int64), None, np.array(16, np.int64)], tol=1e-5)
add("STFT_window", "STFT", 17, [f(1, 50, 1), np.array(5, np.int64), f(12, lo=0, hi=1)], tol=1e-5)
for op in ["HannWindow", "HammingWindow", "BlackmanWindow"]:
    for per in [0, 1]:
        add(f"{op}_{per}", op, 17, [np.array(11, np.int64)], {"periodic": per})
for ac in [0, 1]:
    add(f"AffineGrid_2d_{ac}", "AffineGrid", 20, [f(2, 2, 3), np.array([2, 1, 4, 5], np.int64)], {"align_corners": ac})
    add(f"AffineGrid_3d_{ac}", "AffineGrid", 20, [f(2, 3, 4), np.array([2, 1, 3, 4, 5], np.int64)], {"align_corners": ac})
for mode in ["linear", "nearest", "cubic"]:
    for pad in ["zeros", "border", "reflection"]:
        for ac in [0, 1]:
            add(f"GridSample_{mode}_{pad}_{ac}", "GridSample", 20, [f(1, 2, 5, 6), f(1, 3, 4, 2, lo=-1.3, hi=1.3)],
                {"mode": mode, "padding_mode": pad, "align_corners": ac})
for mode in ["linear", "nearest"]:
    add(f"GridSample_3d_{mode}", "GridSample", 20, [f(1, 2, 4, 5, 3), f(1, 2, 3, 2, 3, lo=-1.2, hi=1.2)], {"mode": mode})
boxes = np.array([[[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9], [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]], np.float32)
scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3], [0.4, 0.8, 0.2, 0.7, 0.9, 0.1]]], np.float32)
add("NMS", "NonMaxSuppression", 11, [boxes, scores, np.array([3], np.int64), np.array([0.5], np.float32), np.array([0.0], np.float32)], tol=0)
add("NMS_limit", "NonMaxSuppression", 11, [boxes, scores, np.array([1], np.int64), np.array([0.5], np.float32), np.array([0.0], np.float32)], tol=0)
add("NMS_center", "NonMaxSuppression", 11, [boxes, scores, np.array([2], np.int64), np.array([0.3], np.float32), np.array([0.5], np.float32)], {"center_point_box": 1}, tol=0)
for mode in ["avg", "max"]:
    for ct in ["half_pixel", "output_half_pixel"]:
        add(f"RoiAlign_{mode}_{ct}", "RoiAlign", 16, [f(2, 3, 8, 8), np.array([[0.5, 1, 6, 5.5], [2, 2, 7, 7]], np.float32), np.array([0, 1], np.int64)],
            {"mode": mode, "coordinate_transformation_mode": ct, "output_height": 3, "output_width": 4, "sampling_ratio": 2, "spatial_scale": 1.0})
add("RoiAlign_auto_ratio", "RoiAlign", 16, [f(1, 2, 9, 9), np.array([[1, 1, 8, 7]], np.float32), np.array([0], np.int64)],
    {"output_height": 2, "output_width": 2, "spatial_scale": 0.5})

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "testdata", "ops-coverage")
os.makedirs(out_dir, exist_ok=True)
text = json.dumps({"oracle": f"onnxruntime {ort.__version__}", "onnx": onnx.__version__, "seed": 20261030, "cases": cases}, separators=(",", ":"))
with open(os.path.join(out_dir, "cases.json.gz"), "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as fh:
    fh.write(text.encode("utf-8"))
print(f"{len(cases)} cases -> {out_dir}", file=sys.stderr)
