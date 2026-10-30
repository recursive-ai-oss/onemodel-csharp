#!/usr/bin/env python3
"""Dumps a checkpoint's weights and the validation line sets in raw form, for a trainer that cannot read PyTorch or PNG files.

Run from the root of a (copy of a) training tree:  python dump_inputs.py CONFIG.yaml CHECKPOINT.pt OUT_DIR

OUT_DIR gets weights.json + weights.bin (state-dict order; float32; running statistics included, batch counters not),
charset.txt, and set-<name>.bin for `val` and every `extra_val` set, limited to the first 1500 (val) or 3000 lines like the trainer's
own evaluation. Set layout (little-endian int32): N, then per line: width, label length, labels[len], and 32*width bytes (row-major).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from data import charset as cs
from train.data import LineSet


def main() -> None:
    cfgp, ckp, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(os.path.expandvars(json.dumps(yaml.safe_load(Path(cfgp).read_text(encoding="utf-8")))))
    charset = cs.load(Path(cfg["charset"]))
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    meta, blob, off = [], bytearray(), 0
    for k, v in ck["model"].items():
        if k.endswith("num_batches_tracked"):
            continue
        a = v.detach().cpu().float().contiguous().numpy()
        meta.append({"name": k, "shape": list(a.shape), "offset": off})
        blob += a.astype("<f4").tobytes()
        off += a.size
    (out / "weights.json").write_text(json.dumps({"config": ck["config"], "tensors": meta}), encoding="utf-8")
    (out / "weights.bin").write_bytes(bytes(blob))
    (out / "charset.txt").write_text("\n".join(charset) + "\n", encoding="utf-8")
    sets = {"val": (Path(cfg["data_dir"]) / "val", 1500)} | {k: (Path(v), 3000) for k, v in cfg.get("extra_val", {}).items()}
    for name, (path, limit) in sets.items():
        ls = LineSet(path, charset, limit)
        n = min(len(ls), limit)
        with open(out / f"set-{name}.bin", "wb") as f:
            f.write(np.array([n], dtype="<i4").tobytes())
            for i in range(n):
                im = ls.images[i]
                assert im.shape[0] == 32
                f.write(np.array([im.shape[1], len(ls.targets[i])], dtype="<i4").tobytes())
                f.write(np.array(ls.targets[i], dtype="<i4").tobytes())
                f.write(np.ascontiguousarray(im).tobytes())
        print(name, n)


if __name__ == "__main__":
    main()
