#!/usr/bin/env python3
"""Straightforward CTC decoding references for the tests (docs/ctc.md): greedy, a prefix beam search, and for small sizes
the exact best text by summing over every alignment. numpy and the standard library only; double precision throughout.

usage: ctc_reference.py OUT.json [--seed N]
"""
import argparse
import itertools
import json
import math

import numpy as np


def log_softmax(z):
    z = z.astype(np.float64)
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def greedy(scores, blank, kind):
    t_n, c_n = scores.shape
    lp = log_softmax(scores) if kind == "logits" else scores.astype(np.float64)
    prob = np.exp(lp)
    best = [int(np.argmax(scores[t])) for t in range(t_n)]  # argmax: lowest index on ties
    runs, last, first, rmax = [], blank, 0, 0.0
    for t, b in enumerate(best):
        p = float(prob[t, b])
        if b != last:
            if last != blank:
                runs.append((last, first, t - 1, rmax))
            if b != blank:
                first, rmax = t, p
        elif b != blank and p > rmax:
            rmax = p
        last = b
    if last != blank:
        runs.append((last, first, t_n - 1, rmax))
    conf = sum(r[3] for r in runs) / len(runs) if runs else 0.0
    return {
        "labels": [r[0] for r in runs], "first": [r[1] for r in runs], "last": [r[2] for r in runs],
        "runConfidence": [r[3] for r in runs], "confidence": conf,
        "stepProbability": [float(prob[t, best[t]]) for t in range(t_n)],
    }


def collapse(path, blank):
    out, prev = [], None
    for s in path:
        if s != prev and s != blank:
            out.append(s)
        prev = s
    return tuple(out)


def exact_best(scores, blank, kind):
    lp = log_softmax(scores) if kind == "logits" else scores.astype(np.float64)
    t_n, c_n = lp.shape
    total = {}
    for path in itertools.product(range(c_n), repeat=t_n):
        text = collapse(path, blank)
        p = sum(lp[t, s] for t, s in enumerate(path))
        total[text] = np.logaddexp(total.get(text, -np.inf), p)
    best = max(total.items(), key=lambda kv: kv[1])
    return list(best[0]), float(best[1])


def prefix_beam(scores, blank, kind, width):
    """Hannun et al. 2014, Algorithm 1, written with dictionaries over label tuples."""
    lp = log_softmax(scores) if kind == "logits" else scores.astype(np.float64)
    beams = {(): (0.0, -np.inf)}  # prefix -> (log p_blank, log p_nonblank)
    for t in range(lp.shape[0]):
        nxt = {}

        def add(prefix, pb, pnb):
            ob, onb = nxt.get(prefix, (-np.inf, -np.inf))
            nxt[prefix] = (np.logaddexp(ob, pb), np.logaddexp(onb, pnb))

        for prefix, (pb, pnb) in beams.items():
            total = np.logaddexp(pb, pnb)
            c = prefix[-1] if prefix else None
            add(prefix, total + lp[t, blank], -np.inf)
            if c is not None:
                add(prefix, -np.inf, pnb + lp[t, c])
            for j in range(lp.shape[1]):
                if j == blank:
                    continue
                v = (pb if j == c else total) + lp[t, j]
                if v > -np.inf:
                    add(prefix + (j,), -np.inf, v)
        ranked = sorted(nxt.items(), key=lambda kv: -np.logaddexp(*kv[1]))  # random floats: no ties
        beams = dict(ranked[:width])
    best = max(beams.items(), key=lambda kv: np.logaddexp(*kv[1]))
    return list(best[0]), float(np.logaddexp(*best[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20261029)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cases = []
    for n in range(40):
        small = n < 24
        t_n = int(rng.integers(1, 6)) if small else int(rng.integers(8, 30))
        c_n = int(rng.integers(2, 5)) if small else int(rng.integers(3, 12))
        blank = 0 if n % 4 else int(rng.integers(c_n))
        kind = "logits" if n % 3 else "logprobs"
        scale = float(rng.choice([0.5, 2.0, 6.0]))
        z = (rng.standard_normal((t_n, c_n)) * scale).astype(np.float32)
        if kind == "logprobs":
            z = log_softmax(z).astype(np.float32)
        case = {"steps": t_n, "classes": c_n, "blank": blank, "kind": kind, "scores": [float(v) for v in z.ravel()],
                "greedy": greedy(z, blank, kind), "beams": []}
        if kind == "logprobs":
            # float32 rounding of a log-softmax: renormalising in double would change the reference slightly; the
            # references use the stored values as probabilities, as the library does.
            pass
        if small:
            labels, logp = exact_best(z, blank, kind)
            case["exact"] = {"labels": labels, "logp": logp}
        for w in (1, 2, 3, 8):
            labels, logp = prefix_beam(z, blank, kind, w)
            case["beams"].append({"width": w, "labels": labels, "logp": logp})
        cases.append(case)
    with open(a.out, "w") as f:
        json.dump({"seed": a.seed, "cases": cases}, f)
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
