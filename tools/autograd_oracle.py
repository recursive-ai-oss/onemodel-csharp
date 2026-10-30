#!/usr/bin/env python3
"""Reference gradients from PyTorch (CPU) for two small models, used by the autograd tests.

Builds a conv net (conv-bn-relu-maxpool, conv-bn-relu-avgpool, linear, log-softmax, NLL) and a two-layer transformer encoder
(embedding, multi-head attention with an additive mask, layer norm, GELU feed-forward) from weights drawn with numpy's seeded
generator, runs forward and backward once, and writes weights, inputs, the loss and every gradient to
<out>/<model>.bin (float32, little endian, tensors back to back) plus <out>/<model>.json (name -> shape and float offset).
Names: "in.*" are inputs, "out.*" are results, "grad.<name>" is the gradient of the parameter or input <name>.
PyTorch is used only through its public API, as a black box. Usage: autograd_oracle.py <out-dir>
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SEED = 20261030


def save(path, tensors):
    manifest = {"torch": torch.__version__, "seed": SEED, "tensors": {}}
    chunks = []
    offset = 0
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


def cnn(out):
    rng = np.random.default_rng(SEED)
    w = {
        "x": normal(rng, 4, 1, 12, 12, scale=1.0),
        "c1w": normal(rng, 4, 1, 3, 3), "c1b": normal(rng, 4),
        "bn1g": 1 + normal(rng, 4, scale=0.1), "bn1b": normal(rng, 4, scale=0.1),
        "c2w": normal(rng, 8, 4, 3, 3, scale=0.2), "c2b": normal(rng, 8),
        "bn2g": 1 + normal(rng, 8, scale=0.1), "bn2b": normal(rng, 8, scale=0.1),
        "fcw": normal(rng, 5, 72, scale=0.2), "fcb": normal(rng, 5),
    }
    targets = np.array([2, 0, 4, 1], dtype="int64")
    t = {k: torch.tensor(v, requires_grad=True) for k, v in w.items()}
    rm1, rv1 = torch.zeros(4), torch.ones(4)
    rm2, rv2 = torch.zeros(8), torch.ones(8)
    h = F.conv2d(t["x"], t["c1w"], t["c1b"], padding=1)
    h = F.batch_norm(h, rm1, rv1, t["bn1g"], t["bn1b"], training=True, momentum=0.1, eps=1e-5)
    h = F.max_pool2d(F.relu(h), 2)
    h = F.conv2d(h, t["c2w"], t["c2b"], padding=1)
    h = F.batch_norm(h, rm2, rv2, t["bn2g"], t["bn2b"], training=True, momentum=0.1, eps=1e-5)
    h = F.avg_pool2d(F.relu(h), 2)
    logits = F.linear(h.reshape(4, 72), t["fcw"], t["fcb"])
    loss = F.nll_loss(F.log_softmax(logits, dim=1), torch.tensor(targets))
    loss.backward()
    res = {"in." + k: v for k, v in w.items()}
    res["in.targets"] = targets.astype("float32")
    res["out.loss"] = loss.reshape(())
    res["out.logits"] = logits
    res["out.rm1"], res["out.rv1"], res["out.rm2"], res["out.rv2"] = rm1, rv1, rm2, rv2
    for k, v in t.items():
        res["grad." + k] = v.grad
    save(os.path.join(out, "cnn"), res)


def encoder(out):
    rng = np.random.default_rng(SEED + 1)
    d, heads, ff, vocab, layers, b, n = 16, 2, 32, 11, 2, 2, 5
    w = {"emb": normal(rng, vocab, d, scale=1.0)}
    for i in range(layers):
        p = f"l{i}."
        for nm in ("q", "k", "v", "o"):
            w[p + "w" + nm] = normal(rng, d, d, scale=0.25)
            w[p + "b" + nm] = normal(rng, d, scale=0.1)
        w[p + "ln1g"], w[p + "ln1b"] = 1 + normal(rng, d, scale=0.1), normal(rng, d, scale=0.1)
        w[p + "w1"], w[p + "b1"] = normal(rng, ff, d, scale=0.25), normal(rng, ff, scale=0.1)
        w[p + "w2"], w[p + "b2"] = normal(rng, d, ff, scale=0.25), normal(rng, d, scale=0.1)
        w[p + "ln2g"], w[p + "ln2b"] = 1 + normal(rng, d, scale=0.1), normal(rng, d, scale=0.1)
    ids = rng.integers(0, vocab, size=(b, n)).astype("int64")
    mask = np.where(rng.random((b, n, n)) < 0.25, -4.0, 0.0).astype("float32")
    proj = normal(rng, b, n, d, scale=1.0)
    t = {k: torch.tensor(v, requires_grad=True) for k, v in w.items()}
    maskt = torch.tensor(mask)
    x = F.embedding(torch.tensor(ids), t["emb"])
    dh = d // heads

    def split(z):
        return z.reshape(b, n, heads, dh).permute(0, 2, 1, 3)

    for i in range(layers):
        p = f"l{i}."
        q = split(F.linear(x, t[p + "wq"], t[p + "bq"]))
        k = split(F.linear(x, t[p + "wk"], t[p + "bk"]))
        v = split(F.linear(x, t[p + "wv"], t[p + "bv"]))
        s = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / dh ** 0.5) + maskt[:, None]
        a = torch.matmul(torch.softmax(s, dim=-1), v).permute(0, 2, 1, 3).reshape(b, n, d)
        x1 = F.layer_norm(x + F.linear(a, t[p + "wo"], t[p + "bo"]), (d,), t[p + "ln1g"], t[p + "ln1b"], 1e-5)
        f = F.linear(F.gelu(F.linear(x1, t[p + "w1"], t[p + "b1"])), t[p + "w2"], t[p + "b2"])
        x = F.layer_norm(x1 + f, (d,), t[p + "ln2g"], t[p + "ln2b"], 1e-5)
    loss = (x * torch.tensor(proj)).sum()
    loss.backward()
    res = {"in." + k: v for k, v in w.items()}
    res["in.ids"] = ids.astype("float32")
    res["in.mask"] = mask
    res["in.proj"] = proj
    res["out.loss"] = loss.reshape(())
    res["out.y"] = x
    for k, v in t.items():
        res["grad." + k] = v.grad
    save(os.path.join(out, "encoder"), res)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(1)
    cnn(out)
    encoder(out)
    print("wrote", out, "with torch", torch.__version__)
