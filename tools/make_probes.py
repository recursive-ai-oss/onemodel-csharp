#!/usr/bin/env python3
"""Cuts a transformer encoder export into single-stage probe models, one file per stage and layer.

    make_probes.py encoder.onnx outdir LAYERS      (LAYERS: comma-separated layer numbers, e.g. 4,5,6)

Each probe maps one tapped value to the next (layer norm, attention block, first and second feed-forward matrix, activation, ...), so
an executor's error for that stage alone can be measured from the same, correctly rounded input (GpuReport --probes). Writes
outdir/manifest.txt with lines `file input-tensor output-tensor`. Uses the Whisper encoder's node naming.
"""
import os
import sys

import onnx
import onnx.utils

src, out, layers = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
os.makedirs(out, exist_ok=True)
lines = []
for l in layers:
    p = f"/e/layers.{l}"
    prev = "/e/Add_2_output_0" if l == 0 else f"/e/layers.{l-1}/Add_1_output_0"
    ln1 = f"{p}/self_attn_layer_norm/LayerNormalization_output_0"
    stages = {
        "ln1": (prev, ln1),
        "attn": (ln1, f"{p}/self_attn/out_proj/Add_output_0"),
        "ln2": (f"{p}/Add_output_0", f"{p}/final_layer_norm/LayerNormalization_output_0"),
        "fc1": (f"{p}/final_layer_norm/LayerNormalization_output_0", f"{p}/fc1/Add_output_0"),
        "gelu": (f"{p}/fc1/Add_output_0", f"{p}/activation_fn/Mul_1_output_0"),
        "fc2": (f"{p}/activation_fn/Mul_1_output_0", f"{p}/fc2/Add_output_0"),
        "ffn": (f"{p}/final_layer_norm/LayerNormalization_output_0", f"{p}/fc2/Add_output_0"),
    }
    for name, (i, o) in stages.items():
        f = f"L{l}_{name}.onnx"
        onnx.utils.extract_model(src, os.path.join(out, f), [i], [o])
        lines.append(f"{f} {i} {o}")
open(os.path.join(out, "manifest.txt"), "w").write("\n".join(lines) + "\n")
