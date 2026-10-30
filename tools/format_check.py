"""Checks a converted model against its ONNX source: every initializer's bytes, and the node list.

    format_check.py MODEL.onnx MODEL.onemodel

The OneModel side is read by format_read.py (numpy only, from docs/format.md). The ONNX side uses the `onnx` package as a black
box to get each initializer's raw bytes (external data included). Exit code 0 when everything is equal.
"""
from __future__ import annotations

import sys
from collections import Counter

import onnx
from onnx import numpy_helper

import format_read


def onnx_bytes(t: onnx.TensorProto) -> bytes:
    if t.data_type == onnx.TensorProto.STRING:
        return b"".join(t.string_data)
    if t.HasField("raw_data") or t.data_location == onnx.TensorProto.EXTERNAL:
        # load_external_data has already filled raw_data for external tensors.
        return bytes(t.raw_data)
    return numpy_helper.to_array(t).tobytes()  # little-endian host


def main() -> int:
    src = onnx.load(sys.argv[1], load_external_data=True)
    ours = format_read.read(sys.argv[2])
    g = ours.manifest["graph"]
    by_name = {t["name"]: t for t in g["initializers"]}
    bad = 0
    total = 0
    for t in src.graph.initializer:
        mine = by_name.get(t.name)
        if mine is None:
            print("missing initializer", t.name)
            bad += 1
            continue
        if mine["dtype"] != t.data_type or mine["dims"] != list(t.dims):
            print("dtype/dims differ for", t.name)
            bad += 1
            continue
        want = onnx_bytes(t)
        if t.data_type == onnx.TensorProto.STRING:
            got = b"".join(ours.tensor_bytes(mine))
        else:
            got = ours.tensor_bytes(mine).tobytes()
        total += len(want)
        if got != want:
            print("bytes differ for", t.name)
            bad += 1
    if len(by_name) != len(src.graph.initializer):
        print("initializer counts differ", len(by_name), len(src.graph.initializer))
        bad += 1
    if Counter(n.op_type for n in src.graph.node) != Counter(n["op"] for n in g["nodes"]):
        print("operator histograms differ")
        bad += 1
    print(f"{len(src.graph.initializer)} initializers, {total} bytes compared, {len(src.graph.node)} nodes: "
          + ("EQUAL" if bad == 0 else f"{bad} differences"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
