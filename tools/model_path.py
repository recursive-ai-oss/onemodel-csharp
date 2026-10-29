"""Print the model file and reference directory for a model:variant:batch row spec.

    model_path.py resnet50:fp32:1   ->  "<model file> <reference dir>"
"""
from __future__ import annotations

import sys

from common import MODELS, REFERENCE, load_manifest, machine_id

model, variant, batch = sys.argv[1].split(":")
f = load_manifest()["models"][model]["files"][variant]
print(MODELS / f["path"], REFERENCE / machine_id() / model / variant / f"b{batch}")
