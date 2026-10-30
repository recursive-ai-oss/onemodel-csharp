"""Reader for the OneModel model format (docs/format.md), written from that document with numpy only.

    python format_read.py MODEL.onemodel          # prints a summary

As a library: `read(path)` returns a `Model` whose `.manifest` is the parsed JSON and whose `.tensor_bytes(t)` gives the raw
bytes of a manifest Tensor object as a numpy view of the memory-mapped file (no copy). `.tensor_array(t)` returns the typed array
for dtypes numpy knows.
"""
from __future__ import annotations

import base64
import json
import struct
import sys
from collections import Counter

import numpy as np

MAGIC = b"ONEMODEL"
HEADER = 64
ALIGN = 64

# ONNX TensorProto.DataType -> bits per element (docs/format.md 2.5); 8 (string) has no fixed size.
BITS = {1: 32, 6: 32, 12: 32, 2: 8, 3: 8, 9: 8, 17: 8, 18: 8, 19: 8, 20: 8, 24: 8, 4: 16, 5: 16, 10: 16, 16: 16,
        7: 64, 11: 64, 13: 64, 14: 64, 15: 128, 21: 4, 22: 4, 23: 4, 25: 2, 26: 2, 27: 6, 28: 6}
NUMPY = {1: "<f4", 2: "u1", 3: "i1", 4: "<u2", 5: "<i2", 6: "<i4", 7: "<i8", 9: "?", 10: "<f2", 11: "<f8", 12: "<u4", 13: "<u8"}


class FormatError(Exception):
    pass


class Model:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
        if size < HEADER:
            raise FormatError(f"{size} bytes is shorter than the {HEADER}-byte header")
        self.data = np.memmap(path, dtype="u1", mode="r")
        h = bytes(self.data[:HEADER])
        if h[:8] != MAGIC:
            raise FormatError("bad magic")
        (self.version, flags, mo, ml, self.data_offset, self.data_length, flen, reserved) = struct.unpack_from("<IIQQQQQQ", h, 8)
        if self.version != 1:
            raise FormatError(f"version {self.version}")
        if flags or reserved:
            raise FormatError("flags or reserved bytes set")
        if flen != size or mo != HEADER or ml > size - HEADER:
            raise FormatError("header disagrees with the file length")
        if self.data_offset % ALIGN or self.data_offset < HEADER + ml or self.data_offset + self.data_length != size:
            raise FormatError("data region is misplaced")
        self.manifest = json.loads(bytes(self.data[HEADER:HEADER + ml]).decode("utf-8"))
        if self.manifest["format"] != "onemodel" or self.manifest["version"] != 1:
            raise FormatError("manifest format/version")

    def tensors(self):
        """Every Tensor object in the model, depth first: graph initializers, sparse values/indices, attribute tensors."""
        def graph(g):
            yield from g.get("initializers", [])
            for s in g.get("sparse_initializers", []):
                yield s["values"]
                yield s["indices"]
            for n in g.get("nodes", []):
                yield from node(n)

        def node(n):
            for a in n.get("attributes", []):
                v = a.get("value")
                t = a["type"]
                if t == 4:
                    yield v
                elif t == 9:
                    yield from v
                elif t == 5:
                    yield from graph(v)
                elif t == 10:
                    for g in v:
                        yield from graph(g)
                elif t == 11:
                    yield v["values"]
                    yield v["indices"]
                elif t == 12:
                    for s in v:
                        yield s["values"]
                        yield s["indices"]

        yield from graph(self.manifest["graph"])
        for f in self.manifest.get("functions", []):
            for n in f.get("nodes", []):
                yield from node(n)

    def tensor_bytes(self, t) -> np.ndarray:
        if t["dtype"] == 8:
            return np.array([base64.b64decode(s) for s in t["strings"]], dtype=object)
        count = 1
        for d in t["dims"]:
            if d < 0:
                raise FormatError("negative dimension")
            count *= d
        want = (count * BITS[t["dtype"]] + 7) // 8
        off, length = t["data"]["offset"], t["data"]["length"]
        if length != want:
            raise FormatError(f"tensor '{t.get('name', '')}': {length} bytes, shape needs {want}")
        if length and off % ALIGN:
            raise FormatError("tensor offset is not 64-byte aligned")
        if off < 0 or off + length > self.data_length:
            raise FormatError("tensor lies outside the data region")
        start = self.data_offset + off
        return self.data[start:start + length]

    def tensor_array(self, t) -> np.ndarray:
        raw = self.tensor_bytes(t)
        return np.frombuffer(raw, dtype=NUMPY[t["dtype"]]).reshape(t["dims"])


def read(path: str) -> Model:
    return Model(path)


def main() -> int:
    m = read(sys.argv[1])
    g = m.manifest["graph"]
    ops = Counter(n["op"] for n in g["nodes"])
    n_tensors = sum(1 for _ in m.tensors())
    total = sum(t["data"]["length"] for t in m.tensors() if t["dtype"] != 8)
    print(f"{sys.argv[1]}: format version {m.version}, ir_version {m.manifest['ir_version']}, "
          f"{len(g['nodes'])} nodes, {n_tensors} tensors, {total} tensor bytes")
    print("inputs:", [i["name"] for i in g["inputs"]], "outputs:", [o["name"] for o in g["outputs"]])
    print("ops:", dict(ops.most_common(8)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
