#!/usr/bin/env python3
"""Generates small RNN / GRU / LSTM cases and records ONNX Runtime's outputs for them.

Each case is a directory with model.onnx, input_<i>.pb and output_<i>.pb (ONNX TensorProto files, the same layout as the
onnx backend tests, without the test_data_set level). ONNX Runtime is used only as a black box, through its public Python API,
at the version printed in cases.txt.

usage: ops_recurrent_oracle.py OUTDIR [--seed N] [--count N] [--big]
  --big   also writes a bidirectional 2-layer-sized LSTM (hidden 192, seq 80, batch 1) like a text-recognition head.
"""
import argparse
import os
import re
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

GATES = {"RNN": 1, "GRU": 3, "LSTM": 4}


def make_case(rng, op, direction, layout, hidden, inp, seq, batch, *, bias, lens, init_h, init_c, peep, clip,
              input_forget, lbr, acts, scale=0.6):
    dirs = 2 if direction == "bidirectional" else 1
    g = GATES[op]

    def r(*shape):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    x = r(batch, seq, inp) if layout == 1 else r(seq, batch, inp)
    w = r(dirs, g * hidden, inp)
    rr = r(dirs, g * hidden, hidden)
    names = ["X", "W", "R"]
    values = [x, w, rr]
    optional = [("B", r(dirs, 2 * g * hidden) if bias else None),
                ("sequence_lens", lens),
                ("initial_h", None), ("initial_c", None), ("P", None)]
    state_shape = (batch, dirs, hidden) if layout == 1 else (dirs, batch, hidden)
    optional[2] = ("initial_h", r(*state_shape) if init_h else None)
    if op == "LSTM":
        optional[3] = ("initial_c", r(*state_shape) if init_c else None)
        optional[4] = ("P", r(dirs, 3 * hidden) if peep else None)
    else:
        optional = optional[:3]
    in_names = list(names)
    # An omitted optional input before a present one is an empty name.
    last = max([i for i, (_, v) in enumerate(optional) if v is not None], default=-1)
    for i, (n, v) in enumerate(optional):
        if i > last:
            break
        if v is None:
            in_names.append("")
        else:
            in_names.append(n)
            values.append(v)
    out_names = ["Y", "Y_h"] + (["Y_c"] if op == "LSTM" else [])
    attrs = dict(hidden_size=hidden, direction=direction)
    if layout:
        attrs["layout"] = layout
    if clip:
        attrs["clip"] = clip
    if input_forget:
        attrs["input_forget"] = 1
    if lbr:
        attrs["linear_before_reset"] = 1
    if acts:
        # One list entry per direction and activation, each direction with its own alpha and beta values.
        attrs["activations"] = acts[0] * dirs
        if acts[1]:
            attrs["activation_alpha"] = acts[1] * dirs
        if acts[2]:
            attrs["activation_beta"] = acts[2] * dirs
    node = helper.make_node(op, in_names, out_names, **attrs)
    inputs_vi = []
    k = 0
    for n in in_names:
        if n == "":
            continue
        v = values[k]
        k += 1
        et = TensorProto.INT32 if v.dtype == np.int32 else TensorProto.FLOAT
        inputs_vi.append(helper.make_tensor_value_info(n, et, list(v.shape)))
    outs_vi = [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in out_names]
    graph = helper.make_graph([node], f"{op}_case", inputs_vi, outs_vi)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, values, in_names


def to_layout1(model, values, in_names, outs):
    names = [n for n in in_names if n != ""]
    vals = []
    for n, v in zip(names, values):
        if n == "X":
            v = np.ascontiguousarray(v.transpose(1, 0, 2))
        elif n in ("initial_h", "initial_c"):
            v = np.ascontiguousarray(v.transpose(1, 0, 2))
        vals.append(v)
    new_outs = [np.ascontiguousarray(outs[0].transpose(2, 0, 1, 3))] + [np.ascontiguousarray(o.transpose(1, 0, 2)) for o in outs[1:]]
    m = onnx.ModelProto()
    m.CopyFrom(model)
    m.graph.node[0].attribute.append(helper.make_attribute("layout", 1))
    for vi in m.graph.input:
        k = names.index(vi.name)
        for i, d in enumerate(vals[k].shape):
            vi.type.tensor_type.shape.dim[i].dim_value = d
    return m, vals, new_outs


def write_case(path, model, values, in_names, outputs):
    os.makedirs(path, exist_ok=True)
    onnx.save(model, os.path.join(path, "model.onnx"))
    names = [n for n in in_names if n != ""]
    for i, v in enumerate(values):
        t = numpy_helper.from_array(v, names[i])
        with open(os.path.join(path, f"input_{i}.pb"), "wb") as f:
            f.write(t.SerializeToString())
    for i, v in enumerate(outputs):
        with open(os.path.join(path, f"output_{i}.pb"), "wb") as f:
            f.write(numpy_helper.from_array(v, f"out{i}").SerializeToString())


def run_ort(model, values, in_names):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    feeds = {n: v for n, v in zip([n for n in in_names if n != ""], values)}
    return s.run(None, feeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--count", type=int, default=90)
    ap.add_argument("--big", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    made, skipped = [], []
    act_sets = {
        "LSTM": [None, (["Sigmoid", "Tanh", "Tanh"], [], []), (["Relu", "Sigmoid", "Tanh"], [], []),
                 (["HardSigmoid", "Tanh", "Tanh"], [0.3], [0.4]), (["Sigmoid", "LeakyRelu", "Tanh"], [0.2], []),
                 (["HardSigmoid", "LeakyRelu", "ScaledTanh"], [0.3, 0.2, 1.2], [0.4, 0.7])],
        "GRU": [None, (["Sigmoid", "Tanh"], [], []), (["Sigmoid", "Relu"], [], []),
                (["Sigmoid", "Elu"], [0.7], []), (["Sigmoid", "Softsign"], [], [])],
        "RNN": [None, (["Tanh"], [], []), (["Relu"], [], []), (["Sigmoid"], [], []), (["Affine"], [0.5], [0.1]),
                (["ThresholdedRelu"], [0.2], []), (["ScaledTanh"], [1.5], [0.6]), (["Softplus"], [], [])],
    }
    for n in range(a.count):
        op = ["LSTM", "GRU", "RNN"][n % 3]
        direction = ["forward", "reverse", "bidirectional"][rng.integers(3)]
        layout = int(rng.integers(2))
        hidden = int(rng.integers(1, 7))
        inp = int(rng.integers(1, 6))
        seq = int(rng.integers(1, 7))
        batch = int(rng.integers(1, 5))
        lens = None
        if rng.random() < 0.4 and batch > 0:
            lens = rng.integers(0 if rng.random() < 0.4 else 1, seq + 1, size=batch).astype(np.int32)
            lens[rng.integers(batch)] = seq
        opts = dict(
            bias=rng.random() < 0.7, lens=lens, init_h=rng.random() < 0.4, init_c=rng.random() < 0.4,
            peep=op == "LSTM" and rng.random() < 0.4, clip=float(rng.choice([0, 0, 0.5, 2.0])),
            input_forget=False, lbr=op == "GRU" and rng.random() < 0.5,
            acts=act_sets[op][rng.integers(len(act_sets[op]))])
        if op == "LSTM" and n % 9 == 0:
            opts["input_forget"] = True
        # ONNX Runtime's CPU kernels refuse layout 1, so layout-1 cases are the layout-0 case with every tensor transposed
        # the way Operators.md defines layout 1 (batch first), and the expected outputs transposed the same way.
        model, values, in_names = make_case(rng, op, direction, 0, hidden, inp, seq, batch, **opts)
        name = f"{n:03d}_{op.lower()}_{direction}_l{layout}"
        try:
            outs = run_ort(model, values, in_names)
        except Exception as e:  # recorded, not hidden
            msg = re.sub(r"/onnxruntime_src\S*", "", str(e).splitlines()[0])
            skipped.append(f"{name}: {msg[-200:]}")
            continue
        if layout == 1:
            model, values, outs = to_layout1(model, values, in_names, outs)
        write_case(os.path.join(a.outdir, name), model, values, in_names, outs)
        made.append(name)
    if a.big:
        model, values, in_names = make_case(rng, "LSTM", "bidirectional", 0, 192, 256, 80, 1, bias=True, lens=None,
                                            init_h=False, init_c=False, peep=False, clip=0, input_forget=False,
                                            lbr=False, acts=None, scale=0.06)  # small weights: a layer this size is chaotic at 0.6
        outs = run_ort(model, values, in_names)
        write_case(os.path.join(a.outdir, "big_lstm_bidirectional"), model, values, in_names, outs)
        made.append("big_lstm_bidirectional")
    with open(os.path.join(a.outdir, "cases.txt"), "w") as f:
        f.write(f"onnxruntime {ort.__version__}, onnx {onnx.__version__}, seed {a.seed}\n")
        f.write("\n".join(made) + "\n")
        if skipped:
            f.write("# ORT refused:\n" + "\n".join(skipped) + "\n")
    print(len(made), "cases,", len(skipped), "refused by ORT")
    for s in skipped:
        print("  ", s)


if __name__ == "__main__":
    sys.exit(main())
