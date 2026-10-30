#!/usr/bin/env python3
"""Reference training runs and optimizer checks from PyTorch (CPU), used by the training tests.

PyTorch is used only through its public API, as a black box. Everything is seeded and single-threaded. Writes, to <out-dir>:
  mlp, cnn, encoder  initial weights, the data set, the batch order, the per-step loss and the final parameters of a
                     training run (50-100 steps) of an MLP with SGD+momentum, the conv net with AdamW + cosine schedule +
                     gradient-norm clipping, and a two-layer transformer encoder classifier with Adam + linear warm-up
  optim              single-parameter-set optimizer checks (5 steps each, gradients given), gradient clipping and schedules
Each is <name>.bin (float32 little endian, tensors back to back) and <name>.json (name -> shape and float offset), the layout of
tools/autograd_oracle.py. Names: "in.*" inputs and initial weights, "out.*" results.
Usage: training_oracle.py <out-dir>
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SEED = 20261031


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


def batches(rng, n, b, steps):
    order = []
    while len(order) < steps:
        perm = rng.permutation(n)
        for i in range(0, n - b + 1, b):
            order.append(perm[i:i + b])
    return np.array(order[:steps], dtype="float32")


def lin_init(rng, out_f, in_f):
    bound = 1.0 / np.sqrt(in_f)
    return ((rng.random((out_f, in_f)) * 2 - 1) * bound).astype("float32"), ((rng.random(out_f) * 2 - 1) * bound).astype("float32")


def mlp(out):
    rng = np.random.default_rng(SEED)
    n, d, h, c, b, steps = 256, 20, 32, 4, 32, 100
    centres = normal(rng, c, d, scale=1.0)
    y = rng.integers(0, c, size=n)
    x = (centres[y] + normal(rng, n, d, scale=1.5)).astype("float32")
    w1, b1 = lin_init(rng, h, d)
    w2, b2 = lin_init(rng, c, h)
    order = batches(rng, n, b, steps)
    p = {k: torch.tensor(v, requires_grad=True) for k, v in dict(w1=w1, b1=b1, w2=w2, b2=b2).items()}
    opt = torch.optim.SGD(p.values(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    xt, yt = torch.tensor(x), torch.tensor(y)
    losses = []
    for s in range(steps):
        idx = torch.tensor(order[s].astype("int64"))
        logits = F.linear(F.relu(F.linear(xt[idx], p["w1"], p["b1"])), p["w2"], p["b2"])
        loss = F.cross_entropy(logits, yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    res = {"in.x": x, "in.y": y.astype("float32"), "in.order": order,
           "in.w1": w1, "in.b1": b1, "in.w2": w2, "in.b2": b2, "out.loss": np.array(losses, dtype="float32")}
    for k, v in p.items():
        res["out." + k] = v
    save(os.path.join(out, "mlp"), res)


def cnn(out):
    rng = np.random.default_rng(SEED + 1)
    n, b, steps, classes = 64, 8, 60, 5
    labels = rng.integers(0, classes, size=n)
    templates = normal(rng, classes, 1, 12, 12, scale=1.0)
    x = (templates[labels] + normal(rng, n, 1, 12, 12, scale=1.0)).astype("float32")
    w = {
        "c1w": normal(rng, 4, 1, 3, 3), "c1b": normal(rng, 4),
        "bn1g": 1 + normal(rng, 4, scale=0.1), "bn1b": normal(rng, 4, scale=0.1),
        "c2w": normal(rng, 8, 4, 3, 3, scale=0.2), "c2b": normal(rng, 8),
        "bn2g": 1 + normal(rng, 8, scale=0.1), "bn2b": normal(rng, 8, scale=0.1),
        "fcw": normal(rng, classes, 72, scale=0.2), "fcb": normal(rng, classes),
    }
    order = batches(rng, n, b, steps)
    t = {k: torch.tensor(v, requires_grad=True) for k, v in w.items()}
    rm1, rv1, rm2, rv2 = torch.zeros(4), torch.ones(4), torch.zeros(8), torch.ones(8)
    lr0, total, max_norm = 3e-3, steps, 0.5
    opt = torch.optim.AdamW(t.values(), lr=lr0, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=0.0)
    xt, yt = torch.tensor(x), torch.tensor(labels)
    losses, norms = [], []
    for s in range(steps):
        idx = torch.tensor(order[s].astype("int64"))
        h = F.conv2d(xt[idx], t["c1w"], t["c1b"], padding=1)
        h = F.batch_norm(h, rm1, rv1, t["bn1g"], t["bn1b"], training=True, momentum=0.1, eps=1e-5)
        h = F.max_pool2d(F.relu(h), 2)
        h = F.conv2d(h, t["c2w"], t["c2b"], padding=1)
        h = F.batch_norm(h, rm2, rv2, t["bn2g"], t["bn2b"], training=True, momentum=0.1, eps=1e-5)
        h = F.avg_pool2d(F.relu(h), 2)
        logits = F.linear(h.reshape(b, 72), t["fcw"], t["fcb"])
        loss = F.cross_entropy(logits, yt[idx])
        opt.zero_grad()
        loss.backward()
        norms.append(torch.nn.utils.clip_grad_norm_(t.values(), max_norm).item())
        opt.step()
        sched.step()
        losses.append(loss.item())
    res = {"in.x": x, "in.y": labels.astype("float32"), "in.order": order}
    res.update({"in." + k: v for k, v in w.items()})
    res["out.loss"] = np.array(losses, dtype="float32")
    res["out.gradnorm"] = np.array(norms, dtype="float32")
    for k, v in t.items():
        res["out." + k] = v
    res["out.rm1"], res["out.rv1"], res["out.rm2"], res["out.rv2"] = rm1, rv1, rm2, rv2
    save(os.path.join(out, "cnn"), res)


def encoder(out):
    rng = np.random.default_rng(SEED + 2)
    d, heads, ff, vocab, layers, n, seq, b, steps = 16, 2, 32, 11, 2, 96, 6, 16, 80
    ids = rng.integers(0, vocab, size=(n, seq))
    y = (ids == 3).any(axis=1).astype("int64")
    w = {"emb": normal(rng, vocab, d, scale=1.0)}
    for i in range(layers):
        p = f"l{i}."
        for nm in ("q", "k", "v", "o"):
            w[p + "w" + nm], w[p + "b" + nm] = lin_init(rng, d, d)
        w[p + "ln1g"], w[p + "ln1b"] = 1 + normal(rng, d, scale=0.1), normal(rng, d, scale=0.1)
        w[p + "w1"], w[p + "b1"] = lin_init(rng, ff, d)
        w[p + "w2"], w[p + "b2"] = lin_init(rng, d, ff)
        w[p + "ln2g"], w[p + "ln2b"] = 1 + normal(rng, d, scale=0.1), normal(rng, d, scale=0.1)
    w["clsw"], w["clsb"] = lin_init(rng, 2, d)
    order = batches(rng, n, b, steps)
    t = {k: torch.tensor(v, requires_grad=True) for k, v in w.items()}
    peak, warm = 2e-3, 10
    opt = torch.optim.Adam(t.values(), lr=peak)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warm)
    idt, yt = torch.tensor(ids), torch.tensor(y)
    dh = d // heads
    losses = []
    for s in range(steps):
        idx = torch.tensor(order[s].astype("int64"))
        x = F.embedding(idt[idx], t["emb"])

        def split(z):
            return z.reshape(b, seq, heads, dh).permute(0, 2, 1, 3)

        for i in range(layers):
            p = f"l{i}."
            q = split(F.linear(x, t[p + "wq"], t[p + "bq"]))
            k = split(F.linear(x, t[p + "wk"], t[p + "bk"]))
            v = split(F.linear(x, t[p + "wv"], t[p + "bv"]))
            sc = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / dh ** 0.5)
            a = torch.matmul(torch.softmax(sc, dim=-1), v).permute(0, 2, 1, 3).reshape(b, seq, d)
            x1 = F.layer_norm(x + F.linear(a, t[p + "wo"], t[p + "bo"]), (d,), t[p + "ln1g"], t[p + "ln1b"], 1e-5)
            f = F.linear(F.gelu(F.linear(x1, t[p + "w1"], t[p + "b1"])), t[p + "w2"], t[p + "b2"])
            x = F.layer_norm(x1 + f, (d,), t[p + "ln2g"], t[p + "ln2b"], 1e-5)
        logits = F.linear(x.mean(dim=1), t["clsw"], t["clsb"])
        loss = F.cross_entropy(logits, yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    res = {"in.ids": ids.astype("float32"), "in.y": y.astype("float32"), "in.order": order}
    res.update({"in." + k: v for k, v in w.items()})
    res["out.loss"] = np.array(losses, dtype="float32")
    for k, v in t.items():
        res["out." + k] = v
    save(os.path.join(out, "encoder"), res)


def optim(out):
    rng = np.random.default_rng(SEED + 3)
    shapes = {"p": (5, 10), "q": (7,)}
    init = {k: normal(rng, *s, scale=1.0) for k, s in shapes.items()}
    grads = [{k: normal(rng, *s, scale=1.0) for k, s in shapes.items()} for _ in range(5)]
    configs = {
        "sgd_plain": (torch.optim.SGD, dict(lr=0.1)),
        "sgd_momentum": (torch.optim.SGD, dict(lr=0.1, momentum=0.9)),
        "sgd_nesterov_wd": (torch.optim.SGD, dict(lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-2)),
        "sgd_dampening_wd": (torch.optim.SGD, dict(lr=0.05, momentum=0.8, dampening=0.5, weight_decay=1e-3)),
        "adam": (torch.optim.Adam, dict(lr=1e-2)),
        "adam_l2": (torch.optim.Adam, dict(lr=1e-2, weight_decay=0.1)),
        "adam_amsgrad": (torch.optim.Adam, dict(lr=1e-2, amsgrad=True)),
        "adam_betas": (torch.optim.Adam, dict(lr=3e-3, betas=(0.8, 0.95), eps=1e-6)),
        "adamw": (torch.optim.AdamW, dict(lr=1e-2, weight_decay=0.05)),
        "adamw_amsgrad": (torch.optim.AdamW, dict(lr=1e-2, weight_decay=0.05, amsgrad=True)),
    }
    res = {"in.p": init["p"], "in.q": init["q"]}
    for k, g in enumerate(grads):
        res[f"in.grad.p.{k}"], res[f"in.grad.q.{k}"] = g["p"], g["q"]
    for name, (cls, kw) in configs.items():
        t = {k: torch.tensor(v.copy(), requires_grad=True) for k, v in init.items()}
        opt = cls(t.values(), **kw)
        for k, g in enumerate(grads):
            for n, v in t.items():
                v.grad = torch.tensor(g[n].copy())
            opt.step()
            if k in (0, 4):
                res[f"out.{name}.p.{k + 1}"], res[f"out.{name}.q.{k + 1}"] = t["p"].clone(), t["q"].clone()
    # Gradient clipping: global L2 norm, infinity norm, and value clipping, on the first gradient set.
    for label, kw in (("norm_small", dict(max_norm=0.5)), ("norm_large", dict(max_norm=100.0)), ("norm_inf", dict(max_norm=0.7, norm_type=float("inf")))):
        t = {k: torch.tensor(init[k].copy(), requires_grad=True) for k in init}
        for n, v in t.items():
            v.grad = torch.tensor(grads[0][n].copy())
        total = torch.nn.utils.clip_grad_norm_(t.values(), **kw)
        res[f"out.clip.{label}.total"] = np.array([total.item()], dtype="float32")
        res[f"out.clip.{label}.p"], res[f"out.clip.{label}.q"] = t["p"].grad.clone(), t["q"].grad.clone()
    t = {k: torch.tensor(init[k].copy(), requires_grad=True) for k in init}
    for n, v in t.items():
        v.grad = torch.tensor(grads[0][n].copy())
    torch.nn.utils.clip_grad_value_(t.values(), 0.6)
    res["out.clip.value.p"], res["out.clip.value.q"] = t["p"].grad.clone(), t["q"].grad.clone()

    # Learning-rate schedules: the rate used by step i (0-based), 100 steps.
    steps = 100
    base = 0.1

    def rates(make):
        v = torch.nn.Parameter(torch.zeros(1))
        o = torch.optim.SGD([v], lr=base)
        s = make(o)
        out = []
        for _ in range(steps):
            out.append(o.param_groups[0]["lr"])
            o.step()
            s.step()
        return np.array(out, dtype="float64").astype("float32")

    sched = torch.optim.lr_scheduler
    res["out.lr.step"] = rates(lambda o: sched.StepLR(o, step_size=10, gamma=0.5))
    res["out.lr.cosine"] = rates(lambda o: sched.CosineAnnealingLR(o, T_max=steps, eta_min=1e-3))
    res["out.lr.warmup_cosine"] = rates(lambda o: sched.SequentialLR(
        o, [sched.LinearLR(o, start_factor=0.01, end_factor=1.0, total_iters=10), sched.CosineAnnealingLR(o, T_max=90, eta_min=1e-3)],
        milestones=[10]))
    res["out.lr.onecycle"] = rates(lambda o: sched.OneCycleLR(o, max_lr=base, total_steps=steps, cycle_momentum=False))
    save(os.path.join(out, "optim"), res)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(1)
    mlp(out)
    cnn(out)
    encoder(out)
    optim(out)
    print("wrote", out, "with torch", torch.__version__)
