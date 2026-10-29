#!/usr/bin/env python3
"""Applies the int8 metric to `Bar check` logs (see docs/oracles.md and docs/int8.md).

    check_int8.py REFERENCE_NOISE_CSV LOG [LOG ...]

A row passes when our scaled error against the stored ONNX Runtime ORT_ENABLE_ALL output is at most ONNX Runtime's own scaled
difference between ORT_ENABLE_ALL and ORT_DISABLE_ALL for the same model, variant, batch and output (noise.csv, written by
make_reference.py). The line printed for each output also carries the decision agreement: how many rows have the same top-1 index
along the last axis, and the smallest cosine similarity of a row against the reference.
"""
import csv
import re
import sys

noise = {}
for r in csv.DictReader(open(sys.argv[1])):
    noise[(r["model"], r["variant"], r["batch"], r["output"])] = float(r["max_abs_over_max_ref"])

for log in sys.argv[2:]:
    spec = log.rsplit("/", 1)[-1].split(".t")[0]  # model_variant_batch
    for line in open(log):
        m = re.match(r"\s+(\S+) \[([\dx]+)\]: \S+ scaled (\S+) elementwise \S+ top1 (\d+)/(\d+) mincos (\S+)", line)
        if not m:
            continue
        out, _, scaled, agree, rows, cos = m.groups()
        hit = [(k, v) for k, v in noise.items() if k[3] == out and spec.startswith(f"{k[0]}_{k[1]}_{k[2]}")]
        if not hit:
            print(f"{spec} {out}: ours {scaled} (no ORT noise row)  top1 {agree}/{rows} cos {cos}")
            continue
        k, floor = hit[0]
        ok = float(scaled) <= floor
        print(f"{spec} {out}: ours {float(scaled):.3e} vs ORT self-difference {floor:.3e} -> {'PASS' if ok else 'FAIL'}  top1 {agree}/{rows} mincos {cos}")
        break
