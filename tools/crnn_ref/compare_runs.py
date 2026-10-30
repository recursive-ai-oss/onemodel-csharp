#!/usr/bin/env python3
"""Compares a reference fine-tune with ours: per-step loss and gradient-norm differences, and CER on the validation sets.

    compare_runs.py --ref-steps REF/steps.jsonl --ours-steps OURS/ours-steps.jsonl \
                    --ref-metrics REF/metrics.jsonl --ours-eval OURS/ours-eval.jsonl [--label step2000] [--first 1000]

Prints markdown tables. A set passes when the CER differs from the reference by at most 0.05 percentage points absolute
or 5 percent relative.
"""
from __future__ import annotations

import argparse
import json
import statistics


def lines(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-steps", required=True)
    ap.add_argument("--ours-steps", required=True)
    ap.add_argument("--ref-metrics")
    ap.add_argument("--ours-eval")
    ap.add_argument("--label", default="step2000")
    ap.add_argument("--first", type=int, default=1000)
    a = ap.parse_args()
    ref = {r["step"]: r for r in lines(a.ref_steps)}
    ours = {r["step"]: r for r in lines(a.ours_steps)}
    common = sorted(set(ref) & set(ours))
    print(f"steps compared: {len(common)} (ref {len(ref)}, ours {len(ours)})\n")
    for key in ("loss", "grad_norm"):
        d = [abs(ours[s][key] - ref[s][key]) for s in common]
        r = [abs(ours[s][key] - ref[s][key]) / max(abs(ref[s][key]), 1e-12) for s in common]
        f = [s for s in common if s < a.first]
        rf = [abs(ours[s][key] - ref[s][key]) / max(abs(ref[s][key]), 1e-12) for s in f]
        print(f"{key}: max abs {max(d):.3e}, median rel {statistics.median(r):.3e}, max rel {max(r):.3e}; "
              f"first {a.first}: max rel {max(rf):.3e}, median rel {statistics.median(rf):.3e}")
    print("\n| step | ref loss | ours loss | abs diff | rel diff | ref |g| | ours |g| | rel diff |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    marks = [0, 1, 2, 5, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999, 1250, 1500, 1750, 1999]
    for s in marks:
        if s in ref and s in ours:
            lr, lo, gr, go = ref[s]["loss"], ours[s]["loss"], ref[s]["grad_norm"], ours[s]["grad_norm"]
            print(f"| {s} | {lr:.6f} | {lo:.6f} | {abs(lo - lr):.2e} | {abs(lo - lr) / lr:.2e} | {gr:.5f} | {go:.5f} | {abs(go - gr) / gr:.2e} |")
    # windowed means smooth out the per-step noise of a chaotic trajectory
    print("\n| steps | ref mean loss | ours mean loss | rel diff of means |")
    print("|---|---:|---:|---:|")
    for lo_s in range(0, max(common) + 1, 200):
        w = [s for s in common if lo_s <= s < lo_s + 200]
        if w:
            mr = statistics.fmean(ref[s]["loss"] for s in w)
            mo = statistics.fmean(ours[s]["loss"] for s in w)
            print(f"| {lo_s}-{lo_s + len(w) - 1} | {mr:.6f} | {mo:.6f} | {abs(mo - mr) / mr:.2e} |")
    if a.ref_metrics and a.ours_eval:
        rm = [r for r in lines(a.ref_metrics) if "extra_val" in r]
        base = next(r for r in lines(a.ref_metrics) if "baseline" in r)["baseline"]
        fin = rm[-1]
        refcer = {"val": fin["val"]["cer"]} | {k: v["cer"] for k, v in fin["extra_val"].items()}
        oe = {r["set"]: r["cer"] for r in lines(a.ours_eval) if r["label"] == a.label}
        print(f"\nCER at {a.label} (reference step {fin['step']}), percent:\n")
        print("| set | start | reference | ours | abs diff (pp) | rel diff | within 0.05 pp or 5% |")
        print("|---|---:|---:|---:|---:|---:|:--:|")
        for k, rc in refcer.items():
            if k in oe:
                o = oe[k]
                d = abs(o - rc) * 100
                rel = abs(o - rc) / rc if rc else float("inf")
                print(f"| {k} | {base[k] * 100:.3f} | {rc * 100:.3f} | {o * 100:.3f} | {d:.3f} | {rel * 100:.1f}% | {'yes' if d <= 0.05 or rel <= 0.05 else 'NO'} |")


if __name__ == "__main__":
    main()
