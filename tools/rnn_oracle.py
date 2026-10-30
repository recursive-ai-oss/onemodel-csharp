#!/usr/bin/env python3
"""Reference gradients from PyTorch (CPU) for the recurrent, CTC, transposed-convolution and resize operators.

Three small models, from weights drawn with numpy's seeded generator, forward and backward once; weights, inputs, losses and every
gradient go to <out>/<model>.bin (float32, little endian, tensors back to back) plus <out>/<model>.json (name -> shape and float
offset), the format of autograd_oracle.py.
  crnn     conv-relu-pool x2 -> collapse height -> 2-layer bidirectional LSTM -> linear -> log-softmax -> CTC loss (mean, with
           different input and target lengths, a repeated label)
  gru      1-layer bidirectional GRU (batch first, with initial state) on a sequence -> weighted sum
  decoder  conv -> ConvTranspose2d (stride 2) -> ReLU -> ConvTranspose2d (stride 2, pad 1, output padding 1, groups 2) -> nearest
           resize -> conv -> bilinear resize (both align_corners modes) -> weighted sum
CTC loss values are recorded for reduction none / mean / sum, and one impossible alignment with zero_infinity.
PyTorch is used only through its public API, as a black box. Usage: rnn_oracle.py <out-dir>
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
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


def rnn_weights(rng, kind, in_size, hidden, layers, bidirectional, w):
    """Draws weights for torch.nn.LSTM/GRU into w under their PyTorch names and returns the module loaded with them."""
    cls = nn.LSTM if kind == "lstm" else nn.GRU
    gates = 4 if kind == "lstm" else 3
    m = cls(in_size, hidden, num_layers=layers, bidirectional=bidirectional, batch_first=False)
    sd = {}
    dirs = 2 if bidirectional else 1
    for l in range(layers):
        for d in range(dirs):
            sfx = f"_l{l}" + ("_reverse" if d else "")
            isz = in_size if l == 0 else hidden * dirs
            sd["weight_ih" + sfx] = normal(rng, gates * hidden, isz, scale=0.4)
            sd["weight_hh" + sfx] = normal(rng, gates * hidden, hidden, scale=0.4)
            sd["bias_ih" + sfx] = normal(rng, gates * hidden, scale=0.2)
            sd["bias_hh" + sfx] = normal(rng, gates * hidden, scale=0.2)
    w.update({kind + "." + k: v for k, v in sd.items()})
    m.load_state_dict({k: torch.tensor(v) for k, v in sd.items()})
    return m


def crnn(out):
    rng = np.random.default_rng(SEED)
    b, hgt, wid, hidden, classes = 2, 8, 24, 5, 7
    w = {"x": normal(rng, b, 1, hgt, wid, scale=1.0),
         "c1w": normal(rng, 4, 1, 3, 3, scale=0.4), "c1b": normal(rng, 4, scale=0.1),
         "c2w": normal(rng, 6, 4, 3, 3, scale=0.25), "c2b": normal(rng, 6, scale=0.1)}
    lstm = rnn_weights(rng, "lstm", 6, hidden, 2, True, w)
    w["fcw"], w["fcb"] = normal(rng, classes, 2 * hidden, scale=0.4), normal(rng, classes, scale=0.1)
    t = {k: torch.tensor(v, requires_grad=True) for k, v in w.items() if not k.startswith("lstm.")}
    for name, p in lstm.named_parameters():
        w_key = "lstm." + name
        t[w_key] = p
    h = F.max_pool2d(F.relu(F.conv2d(t["x"], t["c1w"], t["c1b"], padding=1)), 2)       # [2,4,4,12]
    h = F.max_pool2d(F.relu(F.conv2d(h, t["c2w"], t["c2b"], padding=1)), (4, 1))       # [2,6,1,12]
    seq = h.squeeze(2).permute(2, 0, 1)                                               # [T=12, B, 6]
    y, _ = lstm(seq)
    lp = F.log_softmax(F.linear(y, t["fcw"], t["fcb"]), dim=2)
    lp.retain_grad()
    in_len = torch.tensor([12, 10])
    tgt_len = torch.tensor([4, 3])
    tgt = torch.tensor([1, 2, 2, 3, 4, 1, 5])
    loss = F.ctc_loss(lp, tgt, in_len, tgt_len, blank=0, reduction="mean")
    loss.backward()
    res = {"in." + k: v for k, v in w.items()}
    res["in.targets"] = tgt.numpy().astype("float32")
    res["in.input_lengths"] = in_len.numpy().astype("float32")
    res["in.target_lengths"] = tgt_len.numpy().astype("float32")
    res["out.loss"] = loss.reshape(())
    res["out.lp"] = lp
    res["grad.lp"] = lp.grad          # note: PyTorch's CTC gradient includes an exp(lp) term; see docs/autograd.md
    for k, v in t.items():
        res["grad." + k] = v.grad
    with torch.no_grad():
        lpd = lp.detach()
        for red in ("none", "sum", "mean"):
            res["out.ctc_" + red] = F.ctc_loss(lpd, tgt, in_len, tgt_len, blank=0, reduction=red).reshape(-1)
        # Sequence 1 is cut to 2 steps for a 3-label target: no alignment; zero_infinity maps the loss to 0.
        short = torch.tensor([12, 2])
        res["out.ctc_inf_none"] = F.ctc_loss(lpd, tgt, short, tgt_len, reduction="none", zero_infinity=False).reshape(-1)
        res["out.ctc_zeroinf_none"] = F.ctc_loss(lpd, tgt, short, tgt_len, reduction="none", zero_infinity=True).reshape(-1)
    save(os.path.join(out, "crnn"), res)


def gru(out):
    rng = np.random.default_rng(SEED + 1)
    t_len, b, isz, hidden = 6, 3, 4, 5
    w = {"x": normal(rng, b, t_len, isz, scale=1.0), "h0": normal(rng, 2, b, hidden, scale=0.5),
         "proj": normal(rng, b, t_len, 2 * hidden, scale=1.0)}
    g = rnn_weights(rng, "gru", isz, hidden, 1, True, w)
    g.batch_first = True
    t = {k: torch.tensor(v, requires_grad=True) for k, v in w.items() if not k.startswith("gru.")}
    for name, p in g.named_parameters():
        t["gru." + name] = p
    y, hn = g(t["x"], t["h0"])
    loss = (y * t["proj"]).sum() + (hn * 0).sum()
    loss.backward()
    res = {"in." + k: v for k, v in w.items()}
    res["out.loss"] = loss.reshape(())
    res["out.y"] = y
    for k, v in t.items():
        if k != "proj":
            res["grad." + k] = v.grad
    save(os.path.join(out, "gru"), res)


def decoder(out):
    rng = np.random.default_rng(SEED + 2)
    w = {"x": normal(rng, 2, 3, 5, 6, scale=1.0),
         "c1w": normal(rng, 4, 3, 3, 3, scale=0.3), "c1b": normal(rng, 4, scale=0.1),
         "t1w": normal(rng, 4, 4, 2, 2, scale=0.4), "t1b": normal(rng, 4, scale=0.1),
         "t2w": normal(rng, 4, 3, 3, 3, scale=0.3), "t2b": normal(rng, 6, scale=0.1),
         "c2w": normal(rng, 2, 6, 3, 3, scale=0.2), "c2b": normal(rng, 2, scale=0.1)}
    t = {k: torch.tensor(v, requires_grad=True) for k, v in w.items()}
    h = F.relu(F.conv2d(t["x"], t["c1w"], t["c1b"], padding=1))                                  # [2,4,5,6]
    h = F.relu(F.conv_transpose2d(h, t["t1w"], t["t1b"], stride=2))                             # [2,4,10,12]
    h = F.relu(F.conv_transpose2d(h, t["t2w"], t["t2b"], stride=2, padding=1, output_padding=1, groups=2))  # [2,6,20,24]
    h = F.interpolate(h, size=(13, 30), mode="nearest")
    h = F.conv2d(h, t["c2w"], t["c2b"], padding=1)                                              # [2,2,13,30]
    a = F.interpolate(h, size=(9, 17), mode="bilinear", align_corners=False)
    c = F.interpolate(h, size=(16, 11), mode="bilinear", align_corners=True)
    pa, pc = normal(rng, *a.shape, scale=1.0), normal(rng, *c.shape, scale=1.0)
    loss = (a * torch.tensor(pa)).sum() + (c * torch.tensor(pc)).sum()
    loss.backward()
    res = {"in." + k: v for k, v in w.items()}
    res["in.pa"], res["in.pc"] = pa, pc
    res["out.loss"] = loss.reshape(())
    res["out.a"], res["out.c"] = a, c
    for k, v in t.items():
        res["grad." + k] = v.grad
    save(os.path.join(out, "decoder"), res)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(1)
    crnn(out)
    gru(out)
    decoder(out)
    print("wrote", out, "with torch", torch.__version__)
