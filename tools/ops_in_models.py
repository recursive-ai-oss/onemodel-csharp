#!/usr/bin/env python3
"""Lists the operator types of the fp32 reference models, with opset, count and the distinct attribute sets, for the operators
named on the command line (default: the convolution / pooling / normalisation / resize family).

Usage: ops_in_models.py [OP ...]     (models come from tools/models.json, under the cache directory)
"""
import collections
import sys

import onnx

from common import MODELS, load_manifest

FAMILY = ["Conv", "ConvTranspose", "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "BatchNormalization", "Resize",
          "Upsample"]


def attr_text(a):
    if a.ints:
        return f"{a.name}={list(a.ints)}"
    if a.type == onnx.AttributeProto.STRING:
        return f"{a.name}={a.s.decode()}"
    if a.type == onnx.AttributeProto.INT:
        return f"{a.name}={a.i}"
    if a.type == onnx.AttributeProto.FLOAT:
        return f"{a.name}={a.f:g}"
    return a.name


def main():
    wanted = set(sys.argv[1:]) or set(FAMILY)
    for model, info in load_manifest()["models"].items():
        f = info["files"].get("fp32")
        if not f:
            continue
        path = MODELS / f["path"]
        if not path.exists():
            print(f"{model}: missing {path}")
            continue
        m = onnx.load(str(path), load_external_data=False)
        opset = {o.domain: o.version for o in m.opset_import}.get("", 0)
        counts = collections.Counter()
        configs = collections.defaultdict(collections.Counter)
        for n in m.graph.node:
            if n.op_type in wanted:
                counts[n.op_type] += 1
                configs[n.op_type][", ".join(sorted(attr_text(a) for a in n.attribute))] += 1
        print(f"== {model} (opset {opset}); all op types: {dict(collections.Counter(n.op_type for n in m.graph.node))}")
        for op, c in counts.items():
            print(f"  {op} x{c}")
            for cfg, k in configs[op].most_common(8):
                print(f"      {k}x  {cfg}")


if __name__ == "__main__":
    main()
