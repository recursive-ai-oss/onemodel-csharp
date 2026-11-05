#!/usr/bin/env python3
"""Reference values for 1-D convolution from PyTorch (CPU), used by the Conv1d tests.

PyTorch is used only through its public API (torch.nn.functional.conv1d, torch.optim.SGD), as a black box. Seeded, single-threaded.
Writes, to <out-dir>, in the layout of tools/autograd_oracle.py (<name>.bin float32 little endian, <name>.json name -> shape, offset):
  grid   a grid of convolutions over stride, padding (ints, "same", "valid", asymmetric "same"), dilation, groups (1, 2, depthwise),
         bias or none and odd lengths. Per case "cN.": in.x, in.w, in.b (if any), in.r (a fixed weighting of the output), out.y,
         and the gradients of sum(y * r): grad.x, grad.w, grad.b. The geometry is in grid.json under "cases".
  train  a small Conv1d classifier trained 20 steps with SGD + momentum: initial weights, data, batch order, per-step loss.
Usage: conv1d_oracle.py <out-dir>
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SEED = 20261105

# (n, cin, length, cout, kernel, stride, padding, dilation, groups, bias)
GRID = [
    (2, 3, 11, 4, 3, 1, 0, 1, 1, True),
    (2, 3, 11, 4, 3, 1, 1, 1, 1, False),
    (1, 2, 7, 5, 1, 1, 0, 1, 1, True),
    (3, 4, 13, 6, 5, 2, 2, 1, 1, True),
    (2, 4, 14, 4, 3, 3, 1, 1, 1, False),
    (2, 3, 12, 4, 3, 1, 2, 2, 1, True),
    (2, 3, 15, 5, 3, 2, 3, 3, 1, True),
    (2, 4, 10, 6, 3, 1, 1, 1, 2, True),
    (2, 6, 9, 4, 2, 2, 0, 1, 2, False),
    (2, 6, 11, 6, 3, 1, 2, 2, 2, True),
    (3, 5, 13, 5, 3, 1, 1, 1, 5, True),
    (2, 8, 17, 8, 5, 2, 2, 1, 8, False),
    (2, 4, 9, 3, 4, 1, "same", 1, 1, True),
    (2, 4, 9, 3, 3, 1, "same", 2, 1, True),
    (1, 3, 8, 4, 2, 1, "same", 1, 1, False),
    (2, 4, 7, 4, 5, 1, "valid", 1, 2, True),
    (1, 1, 5, 1, 5, 1, 0, 1, 1, True),
    (4, 2, 3, 3, 3, 1, 4, 1, 1, True),
]


def save(path, tensors, extra=None):
    manifest = {"torch": torch.__version__, "seed": SEED, "tensors": {}}
    if extra:
        manifest.update(extra)
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


def normal(rng, *shape, scale=0.5):
    return (rng.standard_normal(shape) * scale).astype("float32")


def grid(out):
    rng = np.random.default_rng(SEED)
    res, cases = {}, []
    for i, (n, cin, length, cout, k, stride, pad, dil, groups, bias) in enumerate(GRID):
        x = normal(rng, n, cin, length, scale=1.0)
        w = normal(rng, cout, cin // groups, k)
        b = normal(rng, cout) if bias else None
        tx, tw = torch.tensor(x, requires_grad=True), torch.tensor(w, requires_grad=True)
        tb = torch.tensor(b, requires_grad=True) if bias else None
        y = F.conv1d(tx, tw, tb, stride=stride, padding=pad, dilation=dil, groups=groups)
        r = normal(rng, *y.shape, scale=1.0)
        (y * torch.tensor(r)).sum().backward()
        p = f"c{i}."
        res[p + "in.x"], res[p + "in.w"] = x, w
        if bias:
            res[p + "in.b"] = b
        res[p + "in.r"], res[p + "out.y"] = r, y
        res[p + "grad.x"], res[p + "grad.w"] = tx.grad, tw.grad
        if bias:
            res[p + "grad.b"] = tb.grad
        cases.append({"stride": stride, "padding": pad, "dilation": dil, "groups": groups, "bias": bias})
    save(os.path.join(out, "grid"), res, {"cases": cases})


def train(out):
    rng = np.random.default_rng(SEED + 1)
    n, b, steps, length, classes = 32, 8, 20, 11, 3
    labels = rng.integers(0, classes, size=n)
    templates = normal(rng, classes, 4, length, scale=1.0)
    x = (templates[labels] + normal(rng, n, 4, length, scale=1.0)).astype("float32")
    w = {
        "c1w": normal(rng, 6, 4, 3, scale=0.3), "c1b": normal(rng, 6, scale=0.3),
        "c2w": normal(rng, 6, 3, 3, scale=0.3), "c2b": normal(rng, 6, scale=0.3),
        "fcw": normal(rng, classes, 6, scale=0.3), "fcb": normal(rng, classes, scale=0.3),
    }
    order = np.array([np.random.default_rng(SEED + 2 + s).permutation(n)[:b] for s in range(steps)], dtype="float32")
    t = {k: torch.tensor(v, requires_grad=True) for k, v in w.items()}
    opt = torch.optim.SGD(t.values(), lr=0.05, momentum=0.9)
    xt, yt = torch.tensor(x), torch.tensor(labels)
    losses = []
    for s in range(steps):
        idx = torch.tensor(order[s].astype("int64"))
        h = F.relu(F.conv1d(xt[idx], t["c1w"], t["c1b"], padding="same", dilation=2))
        h = F.relu(F.conv1d(h, t["c2w"], t["c2b"], stride=2, padding=1, groups=2))
        logits = F.linear(h.mean(dim=2), t["fcw"], t["fcb"])
        loss = F.cross_entropy(logits, yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    res = {"in.x": x, "in.y": labels.astype("float32"), "in.order": order}
    res.update({"in." + k: v for k, v in w.items()})
    res["out.loss"] = np.array(losses, dtype="float32")
    for k, v in t.items():
        res["out." + k] = v
    save(os.path.join(out, "train"), res)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(1)
    grid(out)
    train(out)
