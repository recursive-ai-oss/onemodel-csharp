#!/usr/bin/env python3
"""Small graphs and PyTorch reference values for training an imported ONNX graph (testdata/finetune).

Three graphs, each with the loss L = sum(output * W) for a seeded weight tensor W, so dL/doutput = W:

  mini_modern   a ModernBERT-style encoder written in PyTorch and exported at opset 17 with the TorchScript-based exporter: token
                embedding, bias-free layer norm, fused QKV projection, rotary embeddings computed in the graph from a frequency
                buffer, a local-window attention mask combined with the padding mask, GeGLU feed-forward, final norm.
  mini_bert     a BERT-style encoder: word and position embeddings, layer norm with bias, post-norm layers with biased projections,
                the additive (1 - mask) * -10000 mask, erf GELU, a tanh pooler (Gemm).
  ops_zoo       a graph written node by node with onnx.helper that uses every operator the trainable path supports (opset 20),
                evaluated by a small PyTorch interpreter of those operators so the reference is torch autograd of the same function.

For each graph the script writes <out>/<name>.onnx, <name>.bin (float32 tensors back to back) and <name>.json (name -> shape and float
offset). Names: "in.*" inputs (integer inputs stored as float), "out.*" results, "w" the loss weights, "grad.<initializer>" the gradient
of each trainable initializer, named as in the ONNX file. PyTorch is used through its public API only. Usage: finetune_oracle.py <out-dir>
"""
import json
import os
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper

SEED = 20261105
OPSET_EXPORT = 17


def save(path, tensors):
    manifest = {"torch": torch.__version__, "onnx": onnx.__version__, "seed": SEED, "tensors": {}}
    chunks, offset = [], 0
    for name, t in tensors.items():
        a = np.ascontiguousarray(t.detach().numpy() if hasattr(t, "detach") else t, dtype="<f4")
        manifest["tensors"][name] = {"shape": list(a.shape), "offset": offset}
        chunks.append(a.reshape(-1))
        offset += a.size
    np.concatenate(chunks).tofile(path + ".bin")
    with open(path + ".json", "w") as f:
        json.dump(manifest, f, indent=1)


def normal(rng, *shape, scale=0.3):
    return (rng.standard_normal(shape) * scale).astype("float32")


def match_initializers(model, params):
    """Maps each trainable ONNX initializer to the torch parameter it holds (equal, or transposed by the exporter)."""
    found = {}
    for init in model.graph.initializer:
        a = numpy_helper.to_array(init)
        if a.dtype != np.float32 or a.size <= 1:
            continue
        for pname, p in params:
            v = p.detach().numpy()
            if v.shape == a.shape and np.array_equal(v, a):
                found[init.name] = (p, False)
                break
            if v.ndim == 2 and v.T.shape == a.shape and np.array_equal(v.T, a):
                found[init.name] = (p, True)
                break
    return found


# ---------------------------------------------------------------------------------------------------------------- mini_modern


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class ModernBlock(nn.Module):
    def __init__(self, d, heads, ff, first, theta):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.attn_norm = nn.Identity() if first else nn.LayerNorm(d, bias=False)
        self.wqkv = nn.Linear(d, 3 * d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.mlp_norm = nn.LayerNorm(d, bias=False)
        self.wi = nn.Linear(d, 2 * ff, bias=False)
        self.wo2 = nn.Linear(ff, d, bias=False)
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, self.dh, 2).float() / self.dh)))

    def forward(self, x, mask_add):
        b, s, d = x.shape
        qkv = self.wqkv(self.attn_norm(x)).reshape(b, s, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        pos = torch.arange(s, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.dh ** 0.5) + mask_add
        a = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).reshape(b, s, d)
        x = x + self.wo(a)
        a, g = self.wi(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.wo2(F.gelu(a) * g)


class MiniModern(nn.Module):
    def __init__(self, vocab=37, d=16, heads=2, ff=24, layers=2, window=2):
        super().__init__()
        self.window = window
        self.emb = nn.Embedding(vocab, d)
        self.emb_norm = nn.LayerNorm(d, bias=False)
        self.blocks = nn.ModuleList([ModernBlock(d, heads, ff, i == 0, 10000.0 if i % 2 else 160000.0) for i in range(layers)])
        self.final_norm = nn.LayerNorm(d, bias=False)

    def forward(self, input_ids, attention_mask):
        s = input_ids.shape[1]
        i = torch.arange(s)
        far = (i[:, None] - i[None, :]).abs() > self.window
        pad = attention_mask[:, None, None, :] == 0
        blocked = far[None, None, :, :] | pad
        mask_add = torch.where(blocked, torch.finfo(torch.float32).min, 0.0)
        x = self.emb_norm(self.emb(input_ids))
        for blk in self.blocks:
            x = blk(x, mask_add)
        return self.final_norm(x)


# ---------------------------------------------------------------------------------------------------------------- mini_bert


class MiniBert(nn.Module):
    def __init__(self, vocab=41, d=16, heads=2, ff=32, layers=2, maxpos=16):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.word = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(maxpos, d)
        self.ln0 = nn.LayerNorm(d, eps=1e-12)
        self.q = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.k = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.v = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.o = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d, eps=1e-12) for _ in range(layers)])
        self.f1 = nn.ModuleList([nn.Linear(d, ff) for _ in range(layers)])
        self.f2 = nn.ModuleList([nn.Linear(ff, d) for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d, eps=1e-12) for _ in range(layers)])
        self.pooler = nn.Linear(d, d)

    def split(self, t):
        b, s, _ = t.shape
        return t.reshape(b, s, self.h, self.dh).permute(0, 2, 1, 3)

    def forward(self, input_ids, attention_mask):
        b, s = input_ids.shape
        pos_ids = torch.arange(s).unsqueeze(0).expand(b, s)
        x = self.ln0(self.word(input_ids) + self.pos(pos_ids))
        ext = (1.0 - attention_mask[:, None, None, :].to(torch.float32)) * -10000.0
        for i in range(len(self.q)):
            q, k, v = self.split(self.q[i](x)), self.split(self.k[i](x)), self.split(self.v[i](x))
            p = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.dh ** 0.5) + ext, dim=-1)
            a = torch.matmul(p, v).permute(0, 2, 1, 3).reshape(b, s, -1)
            x = self.ln1[i](x + self.o[i](a))
            x = self.ln2[i](x + self.f2[i](F.gelu(self.f1[i](x))))
        return x, torch.tanh(self.pooler(x[:, 0]))


def export_torch_model(out, name, model, rng):
    model.eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.tensor(normal(rng, *p.shape, scale=0.4 if p.ndim == 1 else 0.25)))
        if hasattr(model, "ln0"):
            for m in model.modules():
                if isinstance(m, nn.LayerNorm):
                    m.weight.add_(1.0)
    ids = torch.tensor(rng.integers(0, 30, size=(2, 7)), dtype=torch.int64)
    mask = torch.ones(2, 7, dtype=torch.int64)
    mask[1, 5:] = 0
    path = os.path.join(out, name + ".onnx")
    with torch.no_grad():
        torch.onnx.export(model, (ids, mask), path, input_names=["input_ids", "attention_mask"],
                          output_names=["hidden"] if name == "mini_modern" else ["hidden", "pooled"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
                          opset_version=OPSET_EXPORT, dynamo=False)
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    outs = model(ids, mask)
    outs = outs if isinstance(outs, tuple) else (outs,)
    ws = [torch.tensor(normal(rng, *o.shape, scale=1.0)) for o in outs]
    loss = sum((o * w).sum() for o, w in zip(outs, ws))
    named = list(model.named_parameters())
    for _, p in named:
        p.grad = None
    loss.backward()
    found = match_initializers(onnx_model, named)
    res = {"in.input_ids": ids.float(), "in.attention_mask": mask.float()}
    for i, (o, w) in enumerate(zip(outs, ws)):
        res[f"out.{i}"] = o
        res[f"w.{i}"] = w
    for init_name, (p, transposed) in found.items():
        g = p.grad
        res["grad." + init_name] = g.T if transposed else g
    trainable = [i.name for i in onnx_model.graph.initializer
                 if numpy_helper.to_array(i).dtype == np.float32 and numpy_helper.to_array(i).size > 1]
    missing = [n for n in trainable if n not in found]
    print(name, "initializers", len(trainable), "matched", len(found), "unmatched (frozen tables):", missing, file=sys.stderr)
    save(os.path.join(out, name), res)


# ---------------------------------------------------------------------------------------------------------------- ops_zoo


def build_zoo(rng):
    """The operator-zoo graph: every operator the trainable path supports, wired so shapes work and every parameter has a gradient."""
    inits = {
        "E": normal(rng, 10, 6, scale=0.8), "W1": normal(rng, 6, 6, scale=0.4), "b1": normal(rng, 6),
        "g": 1 + normal(rng, 6, scale=0.2), "bln": normal(rng, 6, scale=0.2),
        "Wq": normal(rng, 6, 6, scale=0.4), "vv": normal(rng, 6, scale=0.6),
        "dv": (1.5 + np.abs(normal(rng, 4, scale=0.3))).astype("float32"),
        "ln2g": 1 + normal(rng, 4, 4, scale=0.2), "Wh": normal(rng, 4, 6, scale=0.4),
        "pe": normal(rng, 1, 4, 1, scale=0.5), "pb": normal(rng, 4, 3, 2, scale=0.5),
        "G": normal(rng, 5, 4, scale=0.5), "cg": normal(rng, 5), "G2": normal(rng, 3, 2, scale=0.5), "cg2": normal(rng, 2),
        "two": np.array(2.0, dtype=np.float32), "three": np.array(3.0, dtype=np.float32), "zero": np.array(0.0, dtype=np.float32),
    }
    ints = {
        "idx_s": np.array(1, dtype=np.int64), "i0": np.array([0], dtype=np.int64),
        "s0": np.array([0], dtype=np.int64), "e2": np.array([2], dtype=np.int64), "ax1": np.array([1], dtype=np.int64),
        "s_neg": np.array([-5], dtype=np.int64), "e6": np.array([6], dtype=np.int64), "ax2": np.array([2], dtype=np.int64),
        "st2": np.array([2], dtype=np.int64), "split": np.array([2, 4], dtype=np.int64),
        "rs_0m1": np.array([0, -1], dtype=np.int64), "m1": np.array([-1], dtype=np.int64), "ax12": np.array([1, 2], dtype=np.int64),
        "shape4": np.array([3, 4, 2, 3], dtype=np.int64), "ex": np.array([3, 4, 4], dtype=np.int64),
    }
    N = helper.make_node
    nodes = [
        N("Gather", ["E", "ids"], ["h0"], axis=0),
        N("Slice", ["x", "s0", "e2", "ax1"], ["xs"]),
        N("Concat", ["h0", "xs"], ["h1"], axis=1),
        N("MatMul", ["h1", "W1"], ["h2"]),
        N("Add", ["h2", "b1"], ["h3"]),
        N("LayerNormalization", ["h3", "g", "bln"], ["h4"], axis=-1, epsilon=1e-5),
        N("Gather", ["h4", "idx_s"], ["x1"], axis=1),
        N("Unsqueeze", ["x1", "ax1"], ["x1u"]),
        N("Slice", ["h4", "s_neg", "e6", "ax2", "st2"], ["h4s"]),
        N("Split", ["h4", "split"], ["a", "b"], axis=2),
        N("Sigmoid", ["a"], ["as"]), N("Tanh", ["a"], ["at"]), N("Mul", ["as", "at"], ["a2"]),
        N("Gelu", ["b"], ["bg"], approximate="none"),
        N("Relu", ["b"], ["br"]), N("Sub", ["bg", "br"], ["b2"]),
        N("Neg", ["b2"], ["b3"]), N("Exp", ["b3"], ["b4"]),
        N("Sqrt", ["b4"], ["b5"]), N("Pow", ["b5", "two"], ["b6"]), N("Pow", ["b6", "three"], ["b7"]),
        N("Erf", ["b7"], ["b8"]),
        N("Softmax", ["b8"], ["sm"], axis=1),
        N("LogSoftmax", ["b2"], ["lsm"], axis=-1),
        N("Add", ["sm", "lsm"], ["c0"]),
        N("Div", ["c0", "dv"], ["c1"]),
        N("Identity", ["c1"], ["c2"]), N("Dropout", ["c2"], ["c3"]),
        N("LayerNormalization", ["c3", "ln2g"], ["c4"], axis=-2, epsilon=1e-5),
        N("Greater", ["c4", "zero"], ["pos"]), N("Where", ["pos", "c4", "c0"], ["c5w"]),
        N("Cast", ["pos"], ["posf"], to=TensorProto.FLOAT), N("Mul", ["c5w", "posf"], ["c5a"]),
        N("Add", ["c5a", "c0"], ["c5"]),
        N("Transpose", ["c5"], ["c6"], perm=[0, 2, 1]),
        N("MatMul", ["c6", "Wh"], ["c7"]),
        N("ReduceMean", ["c7", "ax2"], ["rm"], keepdims=0),
        N("ReduceSum", ["c7", "ax12"], ["rsum"], keepdims=1),
        N("Reshape", ["rm", "rs_0m1"], ["rmr"]),
        N("Shape", ["c7"], ["sh"]), N("Gather", ["sh", "i0"], ["n0"], axis=0),
        N("Concat", ["n0", "m1"], ["shp"], axis=0),
        N("Reshape", ["c7", "shp"], ["fl"]),
        N("Flatten", ["c7"], ["fl2"], axis=1),
        N("Add", ["fl", "fl2"], ["fsum"]),
        N("MatMul", ["vv", "Wq"], ["vq"]),
        N("MatMul", ["vq", "vv"], ["vqq"]),
        N("Reshape", ["c7", "shape4"], ["t4"]),
        N("Transpose", ["t4"], ["m4"], perm=[0, 1, 3, 2]),
        N("MatMul", ["t4", "m4"], ["p4"]),
        N("MatMul", ["t4", "pb"], ["pbo"]),
        N("Add", ["p4", "pbo"], ["p5"]),
        N("Expand", ["pe", "ex"], ["pex"]),
        N("Add", ["pex", "c5"], ["c8"]),
        N("Cast", ["c8"], ["c8f"], to=TensorProto.FLOAT),
        N("Gemm", ["rmr", "G", "cg"], ["gm"], alpha=0.7, beta=1.3, transB=1),
        N("Gemm", ["rmr", "G2", "cg2"], ["gm2"], alpha=1.0, beta=0.5, transA=1),
    ]
    outputs = ["c8f", "p5", "gm", "gm2", "fsum", "x1u", "a2", "vqq", "rsum", "h4s"]
    return inits, ints, nodes, outputs


def torch_eval(model, env):
    """Evaluates the supported operators with torch ops, in node order. env maps names to tensors and is extended in place."""
    for n in model.graph.node:
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        i = [env[x] if x else None for x in n.input]
        t = n.op_type
        if t == "Gather":
            data, idx, axis = i[0], i[1], a.get("axis", 0)
            if idx.ndim == 0:
                o = data.select(axis, int(idx))
            else:
                o = data.index_select(axis, idx.flatten()).reshape(*data.shape[:axis], *idx.shape, *data.shape[axis + 1:])
        elif t == "Slice":
            starts, ends, axes = i[1].tolist(), i[2].tolist(), i[3].tolist()
            steps = i[4].tolist() if len(i) > 4 else [1] * len(starts)
            sl = [slice(None)] * i[0].ndim
            for s, e, ax, st in zip(starts, ends, axes, steps):
                sl[ax] = slice(s, e, st)
            o = i[0][tuple(sl)]
        elif t == "Split":
            o = torch.split(i[0], i[1].tolist(), dim=a["axis"])
        elif t == "Concat":
            o = torch.cat(i, dim=a["axis"])
        elif t == "MatMul":
            o = torch.matmul(i[0], i[1])
        elif t == "Gemm":
            x, y = (i[0].T if a.get("transA", 0) else i[0]), (i[1].T if a.get("transB", 0) else i[1])
            o = a.get("alpha", 1.0) * (x @ y) + a.get("beta", 1.0) * i[2]
        elif t == "LayerNormalization":
            ax = a.get("axis", -1)
            o = F.layer_norm(i[0], tuple(i[0].shape[ax:]), i[1], i[2] if len(i) > 2 else None, a.get("epsilon", 1e-5))
        elif t == "Softmax":
            o = torch.softmax(i[0], a.get("axis", -1))
        elif t == "LogSoftmax":
            o = torch.log_softmax(i[0], a.get("axis", -1))
        elif t == "Gelu":
            o = F.gelu(i[0])
        elif t in ("Sigmoid", "Tanh", "Relu", "Exp", "Sqrt", "Erf", "Neg"):
            o = {"Sigmoid": torch.sigmoid, "Tanh": torch.tanh, "Relu": torch.relu, "Exp": torch.exp, "Sqrt": torch.sqrt,
                 "Erf": torch.erf, "Neg": torch.neg}[t](i[0])
        elif t in ("Add", "Sub", "Mul", "Div", "Pow"):
            o = {"Add": torch.add, "Sub": torch.sub, "Mul": torch.mul, "Div": torch.div, "Pow": torch.pow}[t](i[0], i[1])
        elif t == "Greater":
            o = i[0] > i[1]
        elif t == "Where":
            o = torch.where(i[0], i[1], i[2])
        elif t == "Cast":
            o = i[0].to({TensorProto.FLOAT: torch.float32}[a["to"]])
        elif t in ("Identity", "Dropout"):
            o = i[0]
        elif t == "Transpose":
            o = i[0].permute(*a["perm"])
        elif t == "Unsqueeze":
            o = i[0]
            for ax in sorted(i[1].tolist()):
                o = o.unsqueeze(ax)
        elif t == "ReduceMean":
            o = i[0].mean(dim=tuple(i[1].tolist()), keepdim=bool(a.get("keepdims", 1)))
        elif t == "ReduceSum":
            o = i[0].sum(dim=tuple(i[1].tolist()), keepdim=bool(a.get("keepdims", 1)))
        elif t == "Reshape":
            spec = i[1].tolist()
            o = i[0].reshape([i[0].shape[k] if d == 0 else d for k, d in enumerate(spec)])
        elif t == "Flatten":
            ax = a.get("axis", 1)
            o = i[0].reshape(int(np.prod(i[0].shape[:ax])), -1)
        elif t == "Shape":
            o = torch.tensor(list(i[0].shape), dtype=torch.int64)
        elif t == "Expand":
            o = torch.broadcast_to(i[0], torch.broadcast_shapes(tuple(i[0].shape), tuple(i[1].tolist())))
        else:
            raise SystemExit(f"zoo interpreter: no mapping for {t}")
        for name, val in zip(n.output, o if isinstance(o, tuple) else (o,)):
            env[name] = val
    return env


def export_zoo(out, rng):
    inits, ints, nodes, outputs = build_zoo(rng)
    f32, i64 = TensorProto.FLOAT, TensorProto.INT64
    x = normal(rng, 3, 4, 6, scale=0.8)
    ids = np.array([[1, 7], [3, 3], [9, 0]], dtype=np.int64)
    def make_model(shapes):
        graph = helper.make_graph(
            nodes, "ops_zoo",
            [helper.make_tensor_value_info("x", f32, ["b", 4, 6]), helper.make_tensor_value_info("ids", i64, ["b", 2])],
            [helper.make_tensor_value_info(o, f32, shapes[o]) for o in outputs],
            [numpy_helper.from_array(v, k) for k, v in {**inits, **ints}.items()])
        m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
        m.ir_version = 9
        return m

    model = make_model({o: [] for o in outputs})
    params = {k: torch.tensor(v, requires_grad=True) for k, v in inits.items() if v.size > 1}
    env = {k: torch.tensor(v) for k, v in inits.items() if v.size <= 1}
    env.update(params)
    env.update({k: torch.tensor(v) for k, v in ints.items()})
    env["x"], env["ids"] = torch.tensor(x), torch.tensor(ids)
    torch_eval(model, env)
    model = make_model({o: list(env[o].shape) for o in outputs})
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(out, "ops_zoo.onnx"))
    outs = [env[o] for o in outputs]
    ws = [torch.tensor(normal(rng, *o.shape, scale=1.0)) for o in outs]
    loss = sum((o * w).sum() for o, w in zip(outs, ws))
    loss.backward()
    res = {"in.x": x, "in.ids": ids.astype(np.float32)}
    for k, (o, w) in enumerate(zip(outs, ws)):
        res[f"out.{k}"] = o
        res[f"w.{k}"] = w
    for k, p in params.items():
        res["grad." + k] = p.grad
    save(os.path.join(out, "ops_zoo"), res)


def export_unsupported(out):
    """A graph with operators that have no gradient on the path from a weight (Sinh, Atan, Clip): loading it must name all three."""
    f32 = TensorProto.FLOAT
    w = numpy_helper.from_array(np.linspace(-1, 1, 6, dtype=np.float32).reshape(2, 3), "w")
    N = helper.make_node
    nodes = [N("Mul", ["x", "w"], ["m"]), N("Sinh", ["m"], ["s"]), N("Atan", ["s"], ["t"]), N("Clip", ["t"], ["y"]),
             N("Sinh", ["x"], ["free"])]
    graph = helper.make_graph(nodes, "unsupported", [helper.make_tensor_value_info("x", f32, [2, 3])],
                              [helper.make_tensor_value_info("y", f32, [2, 3]), helper.make_tensor_value_info("free", f32, [2, 3])], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(out, "unsupported.onnx"))


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    export_torch_model(out, "mini_modern", MiniModern(), rng)
    export_torch_model(out, "mini_bert", MiniBert(), rng)
    export_zoo(out, rng)
    export_unsupported(out)


if __name__ == "__main__":
    main()
