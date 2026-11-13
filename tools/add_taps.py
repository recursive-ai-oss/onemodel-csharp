#!/usr/bin/env python3
"""Adds every node output whose name matches a regex as a graph output, so an executor's intermediate values can be read.

    add_taps.py in.onnx out.onnx REGEX

Used by GpuReport --attribute to measure error layer by layer against a float64 evaluation.
"""
import re
import sys

import onnx
from onnx import helper

src, dst, pattern = sys.argv[1:4]
m = onnx.load(src)
g = m.graph
rx = re.compile(pattern)
existing = {o.name for o in g.output}
for n in g.node:
    for o in n.output:
        if o and o not in existing and rx.search(o):
            g.output.append(helper.make_tensor_value_info(o, onnx.TensorProto.FLOAT, None))
onnx.save(m, dst)
