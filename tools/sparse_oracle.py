#!/usr/bin/env python3
"""Reference values for frozen parameters and row-sparse embedding gradients, from PyTorch (CPU) used as a black box.

Writes <out-dir>/frozen and <out-dir>/sparse (.bin float32 little endian + .json manifest, the layout of training_oracle.py).
  frozen  the two-layer encoder classifier of training_oracle.py (same seeded data and weights) trained with Adam while the
          embedding and the whole first layer have requires_grad_(False): per-step losses and every parameter after the run.
  sparse  a small embedding classifier (padding id 0, repeated ids) trained for a few steps with
          (a) sparse SGD on the embedding + SGD on the head, (b) SparseAdam on the embedding + AdamW on the head,
          (c) SparseAdam with other betas and a large epsilon; plus the dense form of the first sparse gradient.
Usage: sparse_oracle.py <out-dir>
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_oracle import SEED, batches, lin_init, normal, save  # noqa: E402


def frozen(out):
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
    for k, v in t.items():
        if k == "emb" or k.startswith("l0."):
            v.requires_grad_(False)
    peak, warm = 2e-3, 10
    opt = torch.optim.Adam(t.values(), lr=peak)  # frozen parameters stay in the list: they have no gradient and are skipped
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
    res = {"out.loss": np.array(losses, dtype="float32")}
    for k, v in t.items():
        res["out." + k] = v
    save(os.path.join(out, "frozen"), res)


def sparse(out):
    rng = np.random.default_rng(SEED + 7)
    vocab, d, classes, b, seq, steps = 40, 8, 3, 6, 5, 6
    emb0 = normal(rng, vocab, d, scale=1.0)
    emb0[0] = normal(rng, d, scale=1.0)  # the padding row: looked up, never updated
    hw, hb = lin_init(rng, classes, d)
    ids = rng.integers(0, 12, size=(steps, b, seq))  # few distinct ids, so many repeats within and across steps
    ids[:, :, 0] = 0                                  # padding id in every row
    ids[:, 0, :] = ids[:, 0, :1] * 0 + 5              # one row that is the same id five times
    ids[:, 1, 2:4] = 7
    y = rng.integers(0, classes, size=(steps, b))
    res = {"in.emb": emb0, "in.hw": hw, "in.hb": hb, "in.ids": ids.astype("float32"), "in.y": y.astype("float32")}

    def run(name, make_sparse_opt, make_head_opt, record_grad=False):
        emb = torch.nn.Embedding(vocab, d, padding_idx=0, sparse=True)
        head = torch.nn.Linear(d, classes)
        with torch.no_grad():
            emb.weight.copy_(torch.tensor(emb0))
            head.weight.copy_(torch.tensor(hw))
            head.bias.copy_(torch.tensor(hb))
        o1 = make_sparse_opt(list(emb.parameters()))
        o2 = make_head_opt(list(head.parameters()))
        losses = []
        for s in range(steps):
            idt = torch.tensor(ids[s])
            loss = F.cross_entropy(head(emb(idt).mean(dim=1)), torch.tensor(y[s]))
            o1.zero_grad()
            o2.zero_grad()
            loss.backward()
            if record_grad and s == 0:
                res[f"out.{name}.grad0"] = emb.weight.grad.to_dense().clone()
            o1.step()
            o2.step()
            if s == 0:
                res[f"out.{name}.emb1"] = emb.weight.detach().clone()  # after the first step: checks the optimizer alone
            losses.append(loss.item())
        res[f"out.{name}.loss"] = np.array(losses, dtype="float32")
        res[f"out.{name}.emb"] = emb.weight.detach().clone()
        res[f"out.{name}.hw"] = head.weight.detach().clone()
        res[f"out.{name}.hb"] = head.bias.detach().clone()

    run("sgd", lambda p: torch.optim.SGD(p, lr=0.5), lambda p: torch.optim.SGD(p, lr=0.5), record_grad=True)
    run("sparseadam", lambda p: torch.optim.SparseAdam(p, lr=0.05, eps=1e-3), lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.01))
    run("sparseadam2", lambda p: torch.optim.SparseAdam(p, lr=0.02, betas=(0.8, 0.95), eps=1e-8), lambda p: torch.optim.Adam(p, lr=0.01))
    save(os.path.join(out, "sparse"), res)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(1)
    frozen(out)
    sparse(out)
    print("wrote", out, "with torch", torch.__version__)
